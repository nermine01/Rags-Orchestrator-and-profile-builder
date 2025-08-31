import os, httpx
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("PCM_OLLAMA_MODEL", "llama3:latest")
PCM_OLLAMA_TIMEOUT = float(os.getenv("PCM_OLLAMA_TIMEOUT", "60"))  # seconds
OPENAI_MODEL_FALLBACK = os.getenv("PCM_OPENAI_MODEL", "gpt-4o-mini")

# Some installs honor this; if yours doesn’t, use Admin/Dev Mode
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # silence warning only
# A few builds honor this alias to avoid symlinks:
os.environ["SPEECHBRAIN_LOCAL_CACHE_STRATEGY"] = "copy"
# Also make sure caches are writable:
os.environ.setdefault("TORCH_HOME", r"C:\Users\Asus\.cache\torch")
os.environ.setdefault("PYANNOTE_CACHE", r"C:\Users\Asus\.cache\pyannote")

import json, subprocess, numpy as np, torch, cv2, librosa, pandas as pd
from functools import lru_cache
from datetime import datetime
from collections import Counter
import httpx

from fer import FER
from pyannote.audio import Pipeline as DiarizationPipeline
from transformers import pipeline, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import whisper
from openai import OpenAI
from app.config import (
    HUGGINGFACE_TOKEN, WHISPER_MODEL,
    LLM_BACKEND, OPENAI_API_KEY, OPENAI_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
)

# ===== Constants =====
device = "cuda" if torch.cuda.is_available() else "cpu"
emotion_labels = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
style_labels   = ["directive", "emotive", "passive", "assertive", "supportive", "analytical", "spontaneous", "reserved"]

pcm_prototypes = {
    "Thinker": "logical structured analytical organized",
    "Harmonizer": "warm caring compassionate emotional",
    "Persister": "dedicated principled observant judgmental",
    "Promoter": "adaptable persuasive charming bold",
    "Imaginer": "imaginative reflective introverted quiet",
    "Rebel": "playful reactive spontaneous fun"
}

trait_weights = {
    "Thinker": ["directive", "analytical", "neutral", "calm"],
    "Harmonizer": ["supportive", "emotive", "happy"],
    "Persister": ["directive", "assertive", "angry", "fearful"],
    "Imaginer": ["reserved", "passive", "sad"],
    "Promoter": ["spontaneous", "assertive", "surprised"],
    "Rebel": ["spontaneous", "emotive", "disgust"]
}

# ===== Models =====
@lru_cache(maxsize=1)
def _models():
    return {
        "face_cascade": cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
        "emotion_detector": FER(mtcnn=True),
        "whisper": whisper.load_model(WHISPER_MODEL, device=device),
        "classifier": pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if device == "cuda" else -1),
        "embedder": SentenceTransformer("all-MiniLM-L6-v2", device=device),
        "diarization": DiarizationPipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HUGGINGFACE_TOKEN),
        "feature_extractor": Wav2Vec2FeatureExtractor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"),
        "audio_emotion": Wav2Vec2ForSequenceClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition").to(device),
        "summarizer": pipeline("summarization", model="facebook/bart-large-cnn")
    }

# ===== Utilities =====
def normalize(d):
    return {k: round(float(v) / sum(d.values()), 4) if sum(d.values()) > 0 else 0.0 for k, v in d.items()}

def get_embedding(text, models):
    return models["embedder"].encode([text])[0]

def compute_embedding_similarity(text, models):
    emb = get_embedding(text, models)
    return {k: round(float(cosine_similarity([emb], [get_embedding(v, models)])[0][0]), 4) for k, v in pcm_prototypes.items()}

def compute_custom_trait_score(style_vec, emotion_vec):
    scores = {pcm: sum(style_vec.get(t, 0) + emotion_vec.get(t, 0) for t in traits) for pcm, traits in trait_weights.items()}
    return normalize(scores)

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return float(obj)
    return obj

# ===== LLM Guideline =====
def _call_llm_for_guideline(sys_prompt: str, user_prompt: str) -> str:
    # 1) Try Ollama (health check -> chat -> generate)
    try:
        with httpx.Client(timeout=PCM_OLLAMA_TIMEOUT) as client:
            # health check
            r = client.get(f"{OLLAMA_BASE_URL}/api/tags")
            r.raise_for_status()

            # try /api/chat (Ollama >= 0.1.41)
            try:
                r = client.post(f"{OLLAMA_BASE_URL}/api/chat", json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.3}
                })
                if r.status_code == 404:
                    raise httpx.HTTPStatusError("404 on /api/chat", request=r.request, response=r)
                r.raise_for_status()
                data = r.json()
                content = (data.get("message", {}) or {}).get("content") or data.get("response") or ""
                if content.strip():
                    return content.strip()
            except Exception:
                # fallback to /api/generate (older Ollama)
                r = client.post(f"{OLLAMA_BASE_URL}/api/generate", json={
                    "model": OLLAMA_MODEL,
                    "prompt": f"{sys_prompt}\n\nUser: {user_prompt}\nAssistant:",
                    "stream": False,
                    "options": {"temperature": 0.3}
                })
                r.raise_for_status()
                data = r.json()
                content = data.get("response", "")
                if content.strip():
                    return content.strip()
                raise RuntimeError("Ollama /api/generate returned empty response")
    except Exception as cause:
        # 2) OpenAI fallback (only if key present)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(f"Ollama failed and OPENAI_API_KEY is not set. Cause: {cause}") from cause
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            chat = client.chat.completions.create(
                model=OPENAI_MODEL_FALLBACK,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return chat.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI fallback failed. Cause: {e} (original: {cause})") from e

# ===== Main Pipeline =====
def run_pcm_analysis(video_path: str) -> dict:
    M = _models()

    # 1. extract audio
    audio_path = video_path.replace(".mp4",".wav")
    if not os.path.exists(audio_path):
        subprocess.run(['ffmpeg','-i',video_path,'-f','wav','-ar','16000','-ac','1','-acodec','pcm_s16le',audio_path], check=True)

    # 2. transcription
    transcript = M["whisper"].transcribe(audio_path)["text"]

    # 3. diarization
    diarization = M["diarization"](audio_path)
    speaker_transcripts = {spk:"" for _,_,spk in diarization.itertracks(yield_label=True)}

    words = transcript.split()
    for i,w in enumerate(words):
        t = i*0.5
        for turn,_,spk in diarization.itertracks(yield_label=True):
            if turn.start <= t <= turn.end:
                speaker_transcripts[spk]+=w+" "

    # 4. audio emotions
    waveform,sr = librosa.load(audio_path,sr=16000)
    audio_emotions = {}
    for turn,_,spk in diarization.itertracks(yield_label=True):
        seg_wave = waveform[int(turn.start*sr):int(turn.end*sr)]
        if len(seg_wave) < 400:
            seg_wave = np.pad(seg_wave,(0,400-len(seg_wave)))
        inputs = M["feature_extractor"](seg_wave,sampling_rate=sr,return_tensors="pt",padding=True)
        for k in inputs: inputs[k]=inputs[k].to(device)
        logits = M["audio_emotion"](**inputs).logits
        pred = emotion_labels[torch.argmax(logits).item()]
        audio_emotions.setdefault(spk,[]).append(pred)

    # 5. video emotions
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames/fps if fps>0 else 0
    video_emotions, emotion_report = {}, []
    while cap.isOpened():
        ret,frame = cap.read()
        if not ret: break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = M["face_cascade"].detectMultiScale(gray,1.1,5)
        for (x,y,w,h) in faces:
            face = frame[y:y+h,x:x+w]
            emo,score = M["emotion_detector"].top_emotion(face)
            if emo:
                for turn,_,spk in diarization.itertracks(yield_label=True):
                    if turn.start <= ts <= turn.end:
                        video_emotions.setdefault(spk,[]).append(emo)
                        emotion_report.append([int(ts*1000),spk,emo,round(score,2)])
                        break
    cap.release()

    # 6. aggregate PCM profiles
    profiles, mean_emotions = {}, {}
    for spk in speaker_transcripts:
        combined = video_emotions.get(spk,[])+audio_emotions.get(spk,[])
        emo_vec = normalize(Counter(combined))
        mean_emotions[spk]=emo_vec

        style_result = M["classifier"](speaker_transcripts[spk], candidate_labels=style_labels, multi_label=True)
        style_vec = normalize(dict(zip(style_result["labels"], style_result["scores"])))

        profiles[spk] = {
            "Emotion Vector": emo_vec,
            "Style Vector": style_vec,
            "Trait-Based Score": compute_custom_trait_score(style_vec,emo_vec),
            "Embedding Similarity": compute_embedding_similarity(speaker_transcripts[spk],M)
        }

    # 7. summarization
    summary = transcript if len(transcript.split())<50 else M["summarizer"](transcript,max_length=130,min_length=30,do_sample=False)[0]['summary_text']

    # 8. guideline
    sys = "You are an expert in communication analysis."
    user = "Based on speaker transcripts and PCM profiles, give personalized communication advice."
    guideline = _call_llm_for_guideline(sys,user)

    return {
        "transcript": transcript,
        "profiles": convert_numpy(profiles),
        "scripts": speaker_transcripts,
        "summary": summary,
        "guideline": guideline,
        "emotions": pd.DataFrame(emotion_report,columns=["Time(ms)","Speaker","Emotion","Score"]).to_dict(orient="records"),
        "duration_seconds": round(duration,2),
        "mean_emotions": convert_numpy(mean_emotions)
    }
