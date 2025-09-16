# prof_worker/cli.py
import argparse, json, sys, os, logging
from pcm_worker.pipeline import run_pcm_analysis

def main():
    # keep 3rd-party loggers noisy but on STDERR so stdout stays clean
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to .mp4")
    p.add_argument("--out", required=True, help="Path to write JSON result")
    args = p.parse_args()

    try:
        result = run_pcm_analysis(args.input)
        # Write JSON to file (only!)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
    except Exception as e:
        # print full error to STDERR and return non-zero exit code
        logging.exception("[PCM CLI] Unhandled error")
        print(str(e), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
