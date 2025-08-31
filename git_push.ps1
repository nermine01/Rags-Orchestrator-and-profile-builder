# git_push_clean.ps1
param(
    [string]$Message = "Auto commit"
)

Write-Host "🧹 Cleaning up .venv, caches, and large files..."

# 1️⃣ Ensure .gitignore has the right rules
@"
.venv/
__pycache__/
*.pyc
*.pyo
*.pyd
*.faiss
*.jsonl
*.txt
"@ | Out-File -FilePath ".gitignore" -Encoding utf8 -Append

git add .gitignore

# 2️⃣ Remove cached unwanted files from Git tracking
git rm -r --cached .venv 2>$null
git rm -r --cached **/__pycache__ 2>$null
git rm -r --cached *.pyc 2>$null
git rm -r --cached *.pyo 2>$null
git rm -r --cached *.pyd 2>$null
git rm -r --cached storage/*.faiss 2>$null
git rm -r --cached storage/*.jsonl 2>$null
git rm -r --cached storage/*.txt 2>$null

Write-Host "✅ Cleanup done. Staging remaining changes..."

# 3️⃣ Stage all changes
git add -A

# 4️⃣ Commit changes
git commit -m "$Message"

# 5️⃣ Ensure large files are tracked with Git LFS
# Add any large PDFs or FAISS indexes you want to keep
git lfs track "*.pdf"
git lfs track "storage/*/*.faiss"
git add .gitattributes
git commit -m "Track large PDFs and FAISS indexes with Git LFS" 2>$null

# 6️⃣ Push to the correct branch
$branch = "president"
git push origin $branch --force

Write-Host "🚀 Code safely pushed to branch '$branch'"
