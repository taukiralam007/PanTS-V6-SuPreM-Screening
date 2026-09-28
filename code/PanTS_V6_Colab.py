from google.colab import drive
drive.mount('/content/drive')

!mkdir -p "/content/drive/MyDrive/PanTS_Project/checkpoints"
!mkdir -p "/content/drive/MyDrive/PanTS_Project/results"
!mkdir -p "/content/drive/MyDrive/PanTS_Project/code"
!mkdir -p "/content/drive/MyDrive/PanTS_Project/logs"
!mkdir -p "/content/drive/MyDrive/PanTS_Project/downloads"

!ls -lh "/content/drive/MyDrive/PanTS_Project"

!cp "/content/PanTS/data/PanTSMini_ImageTe_00009001_00009901.tar.gz" \
"/content/drive/MyDrive/PanTS_Project/downloads/"

!nvidia-smi

import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

!nvidia-smi

import os, shutil, torch

print("\nGPU:", torch.cuda.get_device_name(0))
print("CUDA available:", torch.cuda.is_available())

total, used, free = shutil.disk_usage("/")
print(f"\nDisk total: {total/1e9:.1f} GB")
print(f"Disk free:  {free/1e9:.1f} GB")

import shutil

total, used, free = shutil.disk_usage("/")

print(f"Total disk : {total/1024**3:.1f} GB")
print(f"Used disk  : {used/1024**3:.1f} GB")
print(f"Free disk  : {free/1024**3:.1f} GB")

import psutil

ram = psutil.virtual_memory()

print(f"RAM total: {ram.total/1024**3:.1f} GB")
print(f"RAM free : {ram.available/1024**3:.1f} GB")

import shutil

total, used, free = shutil.disk_usage("/")

print(f"Total disk : {total/1024**3:.1f} GB")
print(f"Used disk  : {used/1024**3:.1f} GB")
print(f"Free disk  : {free/1024**3:.1f} GB")

%%bash
set -e

cd /content

# Cloned official PanTS repository
if [ ! -d "PanTS" ]; then
    git clone https://github.com/MrGiovanni/PanTS.git
fi

cd /content/PanTS
mkdir -p data/ImageTe

# Downloaded  the official 901-case test images
wget -c --show-progress \
  -O data/PanTSMini_ImageTe_00009001_00009901.tar.gz \
  "https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/PanTSMini_ImageTe_00009001_00009901.tar.gz?download=true"

!mkdir -p "/content/drive/MyDrive/PanTS_Project/checkpoints"
!mkdir -p "/content/drive/MyDrive/PanTS_Project/results"
!mkdir -p "/content/drive/MyDrive/PanTS_Project/code"
!mkdir -p "/content/drive/MyDrive/PanTS_Project/logs"

!ls -lh "/content/drive/MyDrive/PanTS_Project"

!rsync -avh --progress \
  /content/PanTS/results/ \
  "/content/drive/MyDrive/PanTS_Project/results/"

!df -h /content/drive

!ls -lh /content/PanTS/data/
!df -h /content/drive

!mkdir -p "/content/drive/MyDrive/PanTS_Project/downloads"

!cp -v \
"/content/PanTS/data/PanTSMini_ImageTe_00009001_00009901.tar.gz" \
"/content/drive/MyDrive/PanTS_Project/downloads/"

!ls -lh "/content/drive/MyDrive/PanTS_Project/downloads/"

!sha256sum \
"/content/PanTS/data/PanTSMini_ImageTe_00009001_00009901.tar.gz" \
"/content/drive/MyDrive/PanTS_Project/downloads/PanTSMini_ImageTe_00009001_00009901.tar.gz"

!tar -xzf \
"/content/PanTS/data/PanTSMini_ImageTe_00009001_00009901.tar.gz" \
-C "/content/PanTS/data/ImageTe"

from pathlib import Path

root = Path("/content/PanTS/data/ImageTe")
cases = sorted([p for p in root.iterdir() if p.is_dir()])

print("Extracted cases:", len(cases))
print("First 5:", [p.name for p in cases[:5]])
print("Last 5:", [p.name for p in cases[-5:]])

!rm -f "/content/PanTS/data/PanTSMini_ImageTe_00009001_00009901.tar.gz"

!df -h /content

!find "/content/PanTS/data/ImageTe/PanTS_00009001" \
  -maxdepth 2 -type f | sort

%%bash
set -e

cd /content/PanTS/data

wget -c --show-progress \
  -O PanTSMini_Label.tar.gz \
  "http://www.cs.jhu.edu/~zongwei/dataset/PanTSMini_Label.tar.gz"

ls -lh PanTSMini_Label.tar.gz

!tar -tzf "/content/PanTS/data/PanTSMini_Label.tar.gz" | head -30

import subprocess
from pathlib import Path

archive = "/content/PanTS/data/PanTSMini_Label.tar.gz"
list_file = "/content/PanTS/test_label_members.txt"
out_dir = Path("/content/PanTS/data/LabelTe")
out_dir.mkdir(parents=True, exist_ok=True)

wanted = {
    "segmentations/pancreas.nii.gz",
    "segmentations/pancreatic_lesion.nii.gz",
}

count = 0

with open(list_file, "w") as fout:
    p = subprocess.Popen(
        ["tar", "-tzf", archive],
        stdout=subprocess.PIPE,
        text=True
    )

    for line in p.stdout:
        line = line.strip()
        parts = line.split("/")

        if len(parts) < 3 or not parts[0].startswith("PanTS_"):
            continue

        try:
            case_num = int(parts[0].split("_")[1])
        except ValueError:
            continue

        relative = "/".join(parts[1:])

        if 9001 <= case_num <= 9901 and relative in wanted:
            fout.write(line + "\n")
            count += 1

    p.wait()

print("Selected archive members:", count)
print("Expected:", 901 * 2)

!tar -xzf "/content/PanTS/data/PanTSMini_Label.tar.gz" \
    -C "/content/PanTS/data/LabelTe" \
    -T "/content/PanTS/test_label_members.txt"

from pathlib import Path

root = Path("/content/PanTS/data/LabelTe")

pancreas = list(root.glob("PanTS_*/segmentations/pancreas.nii.gz"))
lesions = list(root.glob("PanTS_*/segmentations/pancreatic_lesion.nii.gz"))

print("Pancreas masks:", len(pancreas))
print("Lesion masks  :", len(lesions))

cases = sorted([p.name for p in root.iterdir() if p.is_dir()])
print("Test cases    :", len(cases))
print("First:", cases[:3])
print("Last :", cases[-3:])

!tar -czf \
"/content/drive/MyDrive/PanTS_Project/downloads/PanTS_Test_Labels_901.tar.gz" \
-C "/content/PanTS/data" LabelTe

!ls -lh "/content/drive/MyDrive/PanTS_Project/downloads/PanTS_Test_Labels_901.tar.gz"

!find "/content/drive/MyDrive" \
-type f \
\( -iname "*.pth" -o -iname "*.pt" -o -iname "*.ckpt" \) \
-print 2>/dev/null

!find "/content/PanTS/data/ImageTe/PanTS_00009001" \
-maxdepth 2 -type f -print

import nibabel as nib
import numpy as np

case = "PanTS_00009001"

ct_path = f"/content/PanTS/data/ImageTe/{case}/ct.nii.gz"
pan_path = f"/content/PanTS/data/LabelTe/{case}/segmentations/pancreas.nii.gz"
les_path = f"/content/PanTS/data/LabelTe/{case}/segmentations/pancreatic_lesion.nii.gz"

ct = nib.load(ct_path)
pan = nib.load(pan_path)
les = nib.load(les_path)

ct_arr = ct.get_fdata()
pan_arr = pan.get_fdata()
les_arr = les.get_fdata()

print("CT shape       :", ct_arr.shape)
print("Pancreas shape :", pan_arr.shape)
print("Lesion shape   :", les_arr.shape)

print("CT range       :", ct_arr.min(), ct_arr.max())
print("Pancreas voxels:", int((pan_arr > 0).sum()))
print("Lesion voxels  :", int((les_arr > 0).sum()))

print("Same shapes    :", ct_arr.shape == pan_arr.shape == les_arr.shape)
print("Same affine CT/Pan:", np.allclose(ct.affine, pan.affine))
print("Same affine CT/Les:", np.allclose(ct.affine, les.affine))

from pathlib import Path
import pandas as pd

rows = []

for n in range(9001, 9902):
    case = f"PanTS_{n:08d}"

    ct = Path(f"/content/PanTS/data/ImageTe/{case}/ct.nii.gz")
    pan = Path(f"/content/PanTS/data/LabelTe/{case}/segmentations/pancreas.nii.gz")
    lesion = Path(f"/content/PanTS/data/LabelTe/{case}/segmentations/pancreatic_lesion.nii.gz")

    rows.append({
        "case": case,
        "ct": str(ct),
        "pancreas": str(pan),
        "lesion": str(lesion),
        "ct_exists": ct.exists(),
        "pancreas_exists": pan.exists(),
        "lesion_exists": lesion.exists(),
    })

df = pd.DataFrame(rows)

print("Cases:", len(df))
print("All CT:", df.ct_exists.all())
print("All pancreas:", df.pancreas_exists.all())
print("All lesion:", df.lesion_exists.all())

save_path = "/content/drive/MyDrive/PanTS_Project/results/test_manifest_901.csv"
df.to_csv(save_path, index=False)

print("Saved:", save_path)

!rm -f "/content/PanTS/data/PanTSMini_ImageTe_00009001_00009901.tar.gz"

df -h /content

!rm -f "/content/PanTS/data/PanTSMini_ImageTe_00009001_00009901.tar.gz"
!df -h /content

%%bash
rm -f "/content/PanTS/data/PanTSMini_ImageTe_00009001_00009901.tar.gz"
df -h /content

%%bash
set -e

mkdir -p /content/PanTS/data/ImageTr
cd /content/PanTS/data

echo "=== Downloading first 1000 PanTS training CTs ==="

wget -c --show-progress \
  -O PanTSMini_ImageTr_00000001_00001000.tar.gz \
  "https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/PanTSMini_ImageTr_00000001_00001000.tar.gz?download=true"

echo
echo "=== Archive size ==="
ls -lh PanTSMini_ImageTr_00000001_00001000.tar.gz

echo
echo "=== Extracting ==="
tar -xzf PanTSMini_ImageTr_00000001_00001000.tar.gz \
  -C /content/PanTS/data/ImageTr

echo
echo "=== Verifying ==="
COUNT=$(find /content/PanTS/data/ImageTr \
  -maxdepth 1 -type d -name 'PanTS_*' | wc -l)

echo "Training cases extracted: $COUNT"

if [ "$COUNT" -eq 1000 ]; then
    echo "1000 cases verified."
    rm -f PanTSMini_ImageTr_00000001_00001000.tar.gz
    echo "Compressed archive deleted to save disk."
else
    echo "WARNING: expected 1000 cases. Archive was NOT deleted."
fi

echo
df -h /content

from google.colab import files
uploaded = files.upload()   # choose PanTS_V4_Colab_AllInOne.py

!mkdir -p "/content/drive/MyDrive/PanTS_Project/logs"

!python -u "/content/PanTS_V4_Colab_AllInOne.py" \
2>&1 | tee -a "/content/drive/MyDrive/PanTS_Project/logs/v4_full_run.log"

!ls -lh "/content/drive/MyDrive/PanTS_Project/v4_suprem/checkpoints/"
!ls -lh "/content/drive/MyDrive/PanTS_Project/v4_suprem/results/"
!ls -lh "/content/drive/MyDrive/PanTS_Project/v4_suprem/Taukir_Alam_PanTS_V4_SuPreM_Submission.zip"

import os

print("Drive mounted:", os.path.exists("/content/drive/MyDrive"))
print("Current /content files:")

from google.colab import drive
drive.mount('/content/drive')

!find "/content/drive/MyDrive" -type f \
\( -iname "*v4*" -o \
   -iname "*suprem*" -o \
   -iname "*.pt" -o \
   -iname "*official*901*" -o \
   -iname "*submission*.zip" \) \
2>/dev/null | head -200

ZIP = "/content/drive/MyDrive/PanTS_Project/v4_suprem/Taukir_Alam_PanTS_V4_SuPreM_Submission.zip"

!ls -lh "$ZIP"
!unzip -l "$ZIP"
!sha256sum "$ZIP"

%%bash

SRC="/content/drive/MyDrive/PanTS_Project/v4_suprem"
REPO="/content/PanTS-V4-SuPreM-Screening"

cd /content
rm -rf "$REPO"

git clone https://github.com/taukiralam007/PanTS-V4-SuPreM-Screening.git
cd "$REPO"

git config user.name "Taukir Alam"
git config user.email "taukir.alam007@gmail.com"

# Main code
cp "$SRC/submission/PanTS_V4_Colab_AllInOne.py" . 2>/dev/null || true

# Best trained model
cp "$SRC/checkpoints/best_v4_suprem_segresnet.pt" . 2>/dev/null || true

# Official 901-case results
cp "$SRC/results/official_901_summary.json" . 2>/dev/null || true
cp "$SRC/results/official_901_cases.csv" . 2>/dev/null || true

# Validation/training results
cp "$SRC/results/v4_validation_cases.csv" . 2>/dev/null || true
cp "$SRC/results/v4_history.csv" ./training_history.csv 2>/dev/null || true
cp "$SRC/results/v4_summary.json" . 2>/dev/null || true

cat > .gitignore <<'EOF'
*.zip
*.tar
*.tar.gz
*.nii
*.nii.gz
__pycache__/
.ipynb_checkpoints/
last_v4_suprem_segresnet.pt
EOF

echo "===== FILES READY FOR GITHUB ====="
ls -lh

git add .
git status
git commit -m "Add PanTS V4 SuPreM model, code, and official 901-case results" || true

from getpass import getpass
token = getpass("GitHub Personal Access Token: ")

import subprocess

repo = "/content/PanTS-V4-SuPreM-Screening"

url = f"https://taukiralam007:{token}@github.com/taukiralam007/PanTS-V4-SuPreM-Screening.git"

subprocess.run(
    ["git", "-C", repo, "remote", "set-url", "origin", url],
    check=True
)

result = subprocess.run(
    ["git", "-C", repo, "push", "-u", "origin", "main"],
    capture_output=True,
    text=True
)

print(result.stdout)
print(result.stderr)

# Remove token from remote immediately
subprocess.run(
    [
        "git", "-C", repo, "remote", "set-url", "origin",
        "https://github.com/taukiralam007/PanTS-V4-SuPreM-Screening.git"
    ]
)

token = None

%%bash
git config --global --unset-all credential.helper 2>/dev/null || true
rm -f ~/.git-credentials

cd /content/PanTS-V4-SuPreM-Screening

git remote set-url origin \
https://github.com/taukiralam007/PanTS-V4-SuPreM-Screening.git

git remote -v

from getpass import getpass
token = getpass("Paste NEW GitHub fine-grained PAT: ")

import subprocess

repo = "/content/PanTS-V4-SuPreM-Screening"

auth_url = (
    f"https://taukiralam007:{token}"
    "@github.com/taukiralam007/PanTS-V4-SuPreM-Screening.git"
)

# Temporarily use authenticated URL
subprocess.run(
    ["git", "-C", repo, "remote", "set-url", "origin", auth_url],
    check=True
)

# Check everything is committed
print(
    subprocess.run(
        ["git", "-C", repo, "status"],
        capture_output=True,
        text=True
    ).stdout
)

# Push
result = subprocess.run(
    ["git", "-C", repo, "push", "-u", "origin", "main"],
    capture_output=True,
    text=True
)

print(result.stdout)
print(result.stderr)

# IMPORTANT: remove token from Git remote
subprocess.run(
    [
        "git", "-C", repo, "remote", "set-url", "origin",
        "https://github.com/taukiralam007/PanTS-V4-SuPreM-Screening.git"
    ],
    check=True
)

token = None

%%bash

sudo apt-get update -qq
sudo apt-get install gh -y -qq

git config --global --unset-all credential.helper 2>/dev/null || true
rm -f ~/.git-credentials

cd /content/PanTS-V4-SuPreM-Screening

git branch --unset-upstream 2>/dev/null || true

git remote set-url origin \
https://github.com/taukiralam007/PanTS-V4-SuPreM-Screening.git

echo "Remote:"
git remote -v

echo
echo "Local commits:"
git log --oneline -5

gh auth login --hostname github.com --git-protocol https --web

!gh auth login --hostname github.com --git-protocol https --web

!gh auth login --hostname github.com --git-protocol https --web

from getpass import getpass
import subprocess

token = getpass("Paste your GitHub token: ")

result = subprocess.run(
    ["gh", "auth", "login", "--hostname", "github.com", "--with-token"],
    input=token + "\n",
    text=True,
    capture_output=True
)

print(result.stdout)
print(result.stderr)

!gh auth status

!gh auth setup-git

%cd /content/PanTS-V4-SuPreM-Screening

!git branch --unset-upstream 2>/dev/null || true
!git remote set-url origin https://github.com/taukiralam007/PanTS-V4-SuPreM-Screening.git
!git push -u origin main

%%bash
cd /content/PanTS-V4-SuPreM-Screening

cat > README.md <<'EOF'
# PanTS V4 SuPreM Screening

PanTS V4 SuPreM technical screening submission.

## Official PanTS Test Evaluation

Evaluation was performed on the official 901-case PanTS test set.

| Metric | Result |
|---|---:|
| Pancreas DSC | 0.8236 |
| Lesion DSC | 0.2995 |
| P-Sen | 0.7483 |
| T-Sen | 0.5901 |
| Specificity | 0.6147 |
| AUC | 0.8087 |

### Evaluation Details

- Total test cases: 901
- Positive cases: 151
- Negative cases: 750
- True Positives: 113
- True Negatives: 461
- False Positives: 289
- False Negatives: 38
- Ground-truth tumors: 161
- Detected ground-truth tumors: 95

## Repository Contents

- `PanTS_V4_Colab_AllInOne.py` — complete training and evaluation pipeline
- `best_v4_suprem_segresnet.pt` — best trained model checkpoint
- `official_901_summary.json` — official test-set summary
- `official_901_cases.csv` — per-case official test results
- `v4_validation_cases.csv` — validation results
- `training_history.csv` — training history
- `v4_summary.json` — V4 experiment summary

## Model

The submitted checkpoint is the best V4 SuPreM SegResNet model used for the official evaluation.

## Author

Taukir Alam, Ph.D.
EOF

git add README.md
git commit -m "Add submission README"
git push

%%bash
cd /content/PanTS-V4-SuPreM-Screening

cat > README.md <<'EOF'
# PanTS V4 SuPreM Screening

PanTS V4 SuPreM technical screening submission.

## Official PanTS Test Evaluation

Evaluation was performed on the official 901-case PanTS test set.

| Metric | Result |
|---|---:|
| Pancreas DSC | 0.8236 |
| Lesion DSC | 0.2995 |
| P-Sen | 0.7483 |
| T-Sen | 0.5901 |
| Specificity | 0.6147 |
| AUC | 0.8087 |

### Evaluation Details

- Total test cases: 901
- Positive cases: 151
- Negative cases: 750
- True Positives: 113
- True Negatives: 461
- False Positives: 289
- False Negatives: 38
- Ground-truth tumors: 161
- Detected ground-truth tumors: 95

## Repository Contents

- `PanTS_V4_Colab_AllInOne.py` — complete training and evaluation pipeline
- `best_v4_suprem_segresnet.pt` — best trained model checkpoint
- `official_901_summary.json` — official test-set summary
- `official_901_cases.csv` — per-case official test results
- `v4_validation_cases.csv` — validation results
- `training_history.csv` — training history
- `v4_summary.json` — V4 experiment summary

## Data Reference

This work was completed as part of a technical screening task provided by Prof. Zongwei Zhou, using the PanTS dataset and reference materials provided through the official PanTS repository.

- Official PanTS repository: https://github.com/MrGiovanni/PanTS
- PanTS: The Pancreatic Tumor Segmentation Dataset

The official PanTS repository includes the PanTS training data and the 901-case in-distribution test set used for evaluation.

No raw medical imaging data are included in this repository.

## Model

The submitted checkpoint is the best V4 SuPreM SegResNet model used for the official evaluation.

## Author

Taukir Alam, Ph.D.
EOF

git add README.md
git commit -m "Add README with PanTS data reference"
git push origin main

%%bash
cd /content/PanTS-V4-SuPreM-Screening

python - <<'PY'
from pathlib import Path

p = Path("README.md")
text = p.read_text()

sentence = "This work was completed as part of a technical screening task provided by Prof. Zongwei Zhou, using the PanTS dataset and reference materials provided through the official PanTS repository."

text = text.replace(sentence, "")
text = text.replace("\n\n\n", "\n\n")

p.write_text(text)
PY

git add README.md
git commit -m "Remove screening task statement from README"
git push origin main

from google.colab import files
uploaded = files.upload()

from pathlib import Path
import shutil
import subprocess

repo = Path("/content/PanTS-V4-SuPreM-Screening")
assets = repo / "assets"
assets.mkdir(exist_ok=True)

# Find uploaded image
uploaded_file = Path("/content/pants_v4_suprem_overview.jpg")

# Put it in assets/
target = assets / "pants_v4_suprem_overview.jpg"
shutil.copy2(uploaded_file, target)

# Update README
readme = repo / "README.md"
text = readme.read_text()

figure = """
![PanTS V4 SuPreM Overview](assets/pants_v4_suprem_overview.jpg)

*Figure 1. Overview of the PanTS V4 SuPreM segmentation pipeline and official 901-case evaluation. The qualitative segmentation example is illustrative; quantitative results are from the completed official evaluation.*

"""

# Add directly below main title, only if not already present
if "assets/pants_v4_suprem_overview.jpg" not in text:
    title = "# PanTS V4 SuPreM Screening\n"
    text = text.replace(title, title + "\n" + figure, 1)
    readme.write_text(text)

# Commit and push
subprocess.run(
    ["git", "-C", str(repo), "add",
     "README.md", "assets/pants_v4_suprem_overview.jpg"],
    check=True
)

subprocess.run(
    ["git", "-C", str(repo), "commit",
     "-m", "Add PanTS V4 SuPreM overview figure"],
    check=False
)

subprocess.run(
    ["git", "-C", str(repo), "push", "origin", "main"],
    check=True
)

print("✅ Image added to GitHub and README updated.")

from pathlib import Path

matches = list(Path("/content").rglob("pants_v4_suprem_overview.jpg"))

print(matches)

from pathlib import Path
import shutil
import subprocess

repo = Path("/content/PanTS-V4-SuPreM-Screening")
source = repo / "pants_v4_suprem_overview.jpg"

assets = repo / "assets"
assets.mkdir(exist_ok=True)

target = assets / "pants_v4_suprem_overview.jpg"

# Move image into assets folder
if source.exists():
    shutil.move(str(source), str(target))

# Update README
readme = repo / "README.md"
text = readme.read_text()

figure = """
![PanTS V4 SuPreM Overview](assets/pants_v4_suprem_overview.jpg)

*Figure 1. Overview of the PanTS V4 SuPreM segmentation pipeline and official 901-case evaluation. The qualitative segmentation example is illustrative; quantitative results are from the completed official evaluation.*

"""

if "assets/pants_v4_suprem_overview.jpg" not in text:
    title = "# PanTS V4 SuPreM Screening\n"
    text = text.replace(title, title + "\n" + figure, 1)
    readme.write_text(text)

# Commit
subprocess.run(
    ["git", "-C", str(repo), "add",
     "README.md",
     "assets/pants_v4_suprem_overview.jpg"],
    check=True
)

commit = subprocess.run(
    ["git", "-C", str(repo), "commit",
     "-m", "Add PanTS V4 SuPreM overview figure"],
    capture_output=True,
    text=True
)

print(commit.stdout)
print(commit.stderr)

# Push
push = subprocess.run(
    ["git", "-C", str(repo), "push", "origin", "main"],
    capture_output=True,
    text=True
)

print(push.stdout)
print(push.stderr)

# ================================================================
# PanTS V5 — SuPreM SegResNet
# Full Colab + Google Drive training / validation pipeline
#
# Main V5 changes:
#   1. Auto-detect ALL available PanTS training cases
#   2. SuPreM pretrained SegResNet
#   3. Pancreas + pancreatic lesion multi-task segmentation
#   4. Strong tumor-positive patient oversampling
#   5. Strong lesion-centered patch sampling
#   6. Tversky + focal BCE lesion loss
#   7. Explicit false-positive penalty on negative patients
#   8. Anatomy-consistency loss
#   9. Mixed precision
#  10. Gradient accumulation
#  11. Resume from Google Drive automatically
#  12. Validation threshold tuning
#  13. Minimum-component tuning
#  14. Pancreas-gate tuning
#  15. P-Sen / T-Sen / Specificity / AUC / DSC evaluation
#  16. Checkpoint selection based on PanTS-style metrics
#
# Taukir Alam
# ================================================================


# ================================================================
# 0. MOUNT GOOGLE DRIVE
# ================================================================

from google.colab import drive
drive.mount("/content/drive")


# ================================================================
# 1. INSTALL
# ================================================================

!pip install -q monai nibabel scipy scikit-learn pandas tqdm


# ================================================================
# 2. IMPORTS
# ================================================================

import os
import glob
import json
import time
import random
import warnings
import subprocess
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import ndimage

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import monai

from tqdm.auto import tqdm

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    SpatialPadd,
    RandCropByLabelClassesd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandGaussianNoised,
    EnsureTyped,
    MapTransform,
)

from monai.data import (
    Dataset,
    DataLoader,
)

from monai.networks.nets import SegResNet

from monai.inferers import sliding_window_inference

from torch.utils.data import WeightedRandomSampler


# ================================================================
# 3. CONFIGURATION
# ================================================================

SEED = 42

# ------------------------------------------------
# Google Drive paths
# ------------------------------------------------

DRIVE_ROOT = "/content/drive/MyDrive"

DATA_ROOT = f"{DRIVE_ROOT}/PanTS_Data"

IMAGE_ROOT = f"{DATA_ROOT}/ImageTr"
LABEL_ROOT = f"{DATA_ROOT}/LabelTr"

PROJECT = f"{DRIVE_ROOT}/PanTS_Project/V5"

CHECKPOINT_DIR = f"{PROJECT}/checkpoints"
RESULT_DIR = f"{PROJECT}/results"
MANIFEST_DIR = f"{PROJECT}/manifests"

for d in [
    PROJECT,
    CHECKPOINT_DIR,
    RESULT_DIR,
    MANIFEST_DIR,
]:
    os.makedirs(d, exist_ok=True)


# ------------------------------------------------
# Training
# ------------------------------------------------

MAX_EPOCHS = 40

PATCH_SIZE = (96, 96, 96)

SPACING = (1.5, 1.5, 1.5)

CT_MIN = -175
CT_MAX = 250

# Positive patients are rare.
# V4 had only ~9% positive patients.
POSITIVE_PATIENT_WEIGHT = 8.0

# Number of samples drawn per epoch.
# None = equal to number of training patients.
MAX_SAMPLES_PER_EPOCH = 3000

# Patch sampling:
# background / pancreas / lesion
#
# Increased lesion probability substantially.
CROP_RATIOS = [1, 3, 12]

NUM_PATCHES_PER_CASE = 1

BATCH_SIZE = 1

GRAD_ACCUM = 2

NUM_WORKERS = 2

BASE_LR = 2e-5
HEAD_LR = 1e-4

WEIGHT_DECAY = 1e-5

EARLY_STOP_PATIENCE = 8


# ------------------------------------------------
# Validation
# ------------------------------------------------

# Every epoch:
FAST_VAL_POSITIVES = 20
FAST_VAL_NEGATIVES = 30

# Full validation periodically.
FULL_VAL_EVERY = 2


# ------------------------------------------------
# Mixed precision
# ------------------------------------------------

USE_AMP = True


# ------------------------------------------------
# SuPreM
# ------------------------------------------------

SUPREM_WEIGHTS = (
    f"{PROJECT}/supervised_suprem_segresnet_2100.pth"
)

SUPREM_URL = (
    "https://huggingface.co/"
    "MrGiovanni/SuPreM/resolve/main/"
    "supervised_suprem_segresnet_2100.pth"
)


# ------------------------------------------------
# Saved models
# ------------------------------------------------

LAST_MODEL = (
    f"{CHECKPOINT_DIR}/last_v5.pt"
)

BEST_MODEL = (
    f"{CHECKPOINT_DIR}/best_v5.pt"
)

HISTORY_FILE = (
    f"{RESULT_DIR}/training_history_v5.csv"
)

BEST_CONFIG_FILE = (
    f"{RESULT_DIR}/best_validation_operating_point.json"
)


# ================================================================
# 4. REPRODUCIBILITY
# ================================================================

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True


if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU is not enabled. "
        "Colab -> Runtime -> Change runtime type -> GPU."
    )

DEVICE = torch.device("cuda")


print("=" * 70)
print("ENVIRONMENT")
print("=" * 70)

print("PyTorch :", torch.__version__)
print("MONAI   :", monai.__version__)
print("GPU     :", torch.cuda.get_device_name(0))

props = torch.cuda.get_device_properties(0)

print(
    "GPU RAM :",
    round(props.total_memory / 1024**3, 1),
    "GB"
)


# ================================================================
# 5. HELPER
# ================================================================

def download_file(url, destination):

    if os.path.exists(destination):
        return

    os.makedirs(
        os.path.dirname(destination),
        exist_ok=True
    )

    print("Downloading:")
    print(url)

    subprocess.check_call([
        "wget",
        "-c",
        "-O",
        destination,
        url,
    ])


# ================================================================
# 6. DISCOVER ALL AVAILABLE TRAINING CASES
# ================================================================

def find_case_files(image_root, label_root):

    image_files = glob.glob(
        f"{image_root}/PanTS_*/ct.nii.gz"
    )

    print("\nCT files discovered:", len(image_files))

    records = []

    for image_path in tqdm(
        sorted(image_files),
        desc="Matching labels"
    ):

        case_id = Path(
            image_path
        ).parent.name

        possible_labels = [

            f"{label_root}/{case_id}/combined_labels.nii.gz",

            f"{label_root}/{case_id}/label.nii.gz",

            f"{label_root}/{case_id}/combined_label.nii.gz",
        ]

        label_path = None

        for p in possible_labels:

            if os.path.exists(p):

                label_path = p
                break

        if label_path is None:
            continue

        records.append({
            "case_id": case_id,
            "image": image_path,
            "label": label_path,
        })

    return pd.DataFrame(records)


master = find_case_files(
    IMAGE_ROOT,
    LABEL_ROOT
)


if len(master) == 0:

    raise RuntimeError(
        "\nNo PanTS image/label pairs were found.\n\n"
        f"Expected CT under:\n{IMAGE_ROOT}/PanTS_*/ct.nii.gz\n\n"
        f"Expected labels under:\n"
        f"{LABEL_ROOT}/PanTS_*/combined_labels.nii.gz"
    )


print("\nMatched image + label cases:", len(master))


# ================================================================
# 7. LABEL AUDIT
#
# PanTS relevant IDs:
#
# 17 pancreas
# 18 pancreas body
# 19 pancreas head
# 20 pancreas tail
# 28 pancreatic lesion
# ================================================================

AUDIT_FILE = (
    f"{MANIFEST_DIR}/master_audit_v5.csv"
)


if os.path.exists(AUDIT_FILE):

    print("\nLoading existing label audit...")

    master = pd.read_csv(
        AUDIT_FILE
    )

else:

    audit_rows = []

    print("\nAuditing labels...")

    for _, row in tqdm(
        master.iterrows(),
        total=len(master)
    ):

        arr = np.asarray(
            nib.load(
                row["label"]
            ).dataobj
        )

        pancreas = np.isin(
            arr,
            [17, 18, 19, 20]
        )

        lesion = (
            arr == 28
        )

        audit_rows.append({

            "case_id":
                row["case_id"],

            "image":
                row["image"],

            "label":
                row["label"],

            "pancreas_voxels":
                int(pancreas.sum()),

            "lesion_voxels":
                int(lesion.sum()),

            "pancreas_valid":
                int(pancreas.any()),

            "has_lesion":
                int(lesion.any()),
        })


    master = pd.DataFrame(
        audit_rows
    )

    master.to_csv(
        AUDIT_FILE,
        index=False
    )


print("\n" + "=" * 70)
print("DATASET AUDIT")
print("=" * 70)

print("Total cases       :", len(master))

print(
    "Tumor positive    :",
    int(master.has_lesion.sum())
)

print(
    "Tumor negative    :",
    int((master.has_lesion == 0).sum())
)

print(
    "Positive rate     :",
    round(
        100 * master.has_lesion.mean(),
        2
    ),
    "%"
)

print(
    "Missing pancreas  :",
    int((master.pancreas_valid == 0).sum())
)


# ================================================================
# 8. TRAIN / VALIDATION SPLIT
#
# If we have full 9K:
#       90 / 10
#
# If only PanTSMini ~1000:
#       80 / 20
#
# Gives more positive validation examples for small dataset.
# ================================================================

if len(master) >= 3000:
    VAL_FRACTION = 0.10
else:
    VAL_FRACTION = 0.20


train_df, val_df = train_test_split(

    master,

    test_size=VAL_FRACTION,

    random_state=SEED,

    stratify=master["has_lesion"]
)


train_df = (
    train_df
    .sort_values("case_id")
    .reset_index(drop=True)
)

val_df = (
    val_df
    .sort_values("case_id")
    .reset_index(drop=True)
)


train_df.to_csv(
    f"{MANIFEST_DIR}/train_v5.csv",
    index=False
)

val_df.to_csv(
    f"{MANIFEST_DIR}/val_v5.csv",
    index=False
)


print("\nTRAIN")
print(
    len(train_df),
    "patients |",
    int(train_df.has_lesion.sum()),
    "tumor positive"
)

print("\nVALIDATION")
print(
    len(val_df),
    "patients |",
    int(val_df.has_lesion.sum()),
    "tumor positive"
)


# ================================================================
# 9. RAW PanTS LABEL -> 3 CLASS
#
# 0 background
# 1 pancreas
# 2 pancreatic lesion
# ================================================================

class PanTSLabelTo3Classd(
    MapTransform
):

    def __init__(
        self,
        keys
    ):

        super().__init__(keys)


    def __call__(
        self,
        data
    ):

        d = dict(data)

        for key in self.keys:

            x = torch.as_tensor(
                d[key]
            )

            x = torch.round(
                x
            ).long()

            output = torch.zeros_like(
                x,
                dtype=torch.long
            )

            pancreas = (
                (x == 17)
                |
                (x == 18)
                |
                (x == 19)
                |
                (x == 20)
            )

            lesion = (
                x == 28
            )

            output[pancreas] = 1

            # lesion takes priority
            output[lesion] = 2

            d[key] = output

        return d


# ================================================================
# 10. 3 CLASS -> TWO BINARY CHANNELS
#
# channel 0:
# pancreas region INCLUDING lesion
#
# channel 1:
# lesion
# ================================================================

class ToMultiTaskChannelsd(
    MapTransform
):

    def __init__(
        self,
        keys
    ):

        super().__init__(keys)


    def __call__(
        self,
        data
    ):

        d = dict(data)

        for key in self.keys:

            x = torch.as_tensor(
                d[key]
            ).long()

            pancreas = (
                x > 0
            ).float()

            lesion = (
                x == 2
            ).float()

            d[key] = torch.cat(
                [
                    pancreas,
                    lesion
                ],
                dim=0
            )

        return d


# ================================================================
# 11. TRANSFORMS
# ================================================================

train_transforms = Compose([

    LoadImaged(
        keys=[
            "image",
            "label"
        ]
    ),

    EnsureChannelFirstd(
        keys=[
            "image",
            "label"
        ]
    ),

    Orientationd(
        keys=[
            "image",
            "label"
        ],
        axcodes="RAS"
    ),

    PanTSLabelTo3Classd(
        keys=["label"]
    ),

    ScaleIntensityRanged(
        keys=["image"],
        a_min=CT_MIN,
        a_max=CT_MAX,
        b_min=0.0,
        b_max=1.0,
        clip=True
    ),

    # after intensity clipping,
    # outside air becomes zero
    CropForegroundd(
        keys=[
            "image",
            "label"
        ],
        source_key="image",
        margin=12,
        allow_smaller=True
    ),

    Spacingd(
        keys=[
            "image",
            "label"
        ],
        pixdim=SPACING,
        mode=(
            "bilinear",
            "nearest"
        )
    ),

    SpatialPadd(
        keys=[
            "image",
            "label"
        ],
        spatial_size=PATCH_SIZE
    ),

    RandCropByLabelClassesd(
        keys=[
            "image",
            "label"
        ],
        label_key="label",
        spatial_size=PATCH_SIZE,
        ratios=CROP_RATIOS,
        num_classes=3,
        num_samples=NUM_PATCHES_PER_CASE,
        allow_smaller=False,
        warn=False
    ),

    ToMultiTaskChannelsd(
        keys=["label"]
    ),

    RandFlipd(
        keys=[
            "image",
            "label"
        ],
        spatial_axis=0,
        prob=0.30
    ),

    RandFlipd(
        keys=[
            "image",
            "label"
        ],
        spatial_axis=1,
        prob=0.30
    ),

    RandFlipd(
        keys=[
            "image",
            "label"
        ],
        spatial_axis=2,
        prob=0.15
    ),

    RandRotate90d(
        keys=[
            "image",
            "label"
        ],
        prob=0.25,
        max_k=3
    ),

    RandShiftIntensityd(
        keys=["image"],
        offsets=0.05,
        prob=0.25
    ),

    RandGaussianNoised(
        keys=["image"],
        prob=0.10,
        mean=0.0,
        std=0.01
    ),

    EnsureTyped(
        keys=[
            "image",
            "label"
        ]
    ),
])


eval_transforms = Compose([

    LoadImaged(
        keys=[
            "image",
            "label"
        ]
    ),

    EnsureChannelFirstd(
        keys=[
            "image",
            "label"
        ]
    ),

    Orientationd(
        keys=[
            "image",
            "label"
        ],
        axcodes="RAS"
    ),

    PanTSLabelTo3Classd(
        keys=["label"]
    ),

    ScaleIntensityRanged(
        keys=["image"],
        a_min=CT_MIN,
        a_max=CT_MAX,
        b_min=0.0,
        b_max=1.0,
        clip=True
    ),

    CropForegroundd(
        keys=[
            "image",
            "label"
        ],
        source_key="image",
        margin=12,
        allow_smaller=True
    ),

    Spacingd(
        keys=[
            "image",
            "label"
        ],
        pixdim=SPACING,
        mode=(
            "bilinear",
            "nearest"
        )
    ),

    ToMultiTaskChannelsd(
        keys=["label"]
    ),

    EnsureTyped(
        keys=[
            "image",
            "label"
        ]
    ),
])


# ================================================================
# 12. DATAFRAME -> MONAI LIST
# ================================================================

def dataframe_to_files(df):

    records = []

    for _, row in df.iterrows():

        records.append({

            "image":
                row["image"],

            "label":
                row["label"],

            "case_id":
                row["case_id"],

            "has_lesion":
                int(
                    row["has_lesion"]
                ),

            "pancreas_valid":
                int(
                    row["pancreas_valid"]
                ),
        })

    return records


train_files = dataframe_to_files(
    train_df
)

val_files = dataframe_to_files(
    val_df
)


# ================================================================
# 13. WEIGHTED TRAINING SAMPLER
# ================================================================

sample_weights = [

    POSITIVE_PATIENT_WEIGHT
    if x["has_lesion"] == 1
    else 1.0

    for x in train_files
]


if MAX_SAMPLES_PER_EPOCH is None:

    samples_per_epoch = len(
        train_files
    )

else:

    samples_per_epoch = min(
        len(train_files),
        MAX_SAMPLES_PER_EPOCH
    )


sampler = WeightedRandomSampler(

    sample_weights,

    num_samples=samples_per_epoch,

    replacement=True
)


train_ds = Dataset(
    train_files,
    transform=train_transforms
)


train_loader = DataLoader(

    train_ds,

    batch_size=BATCH_SIZE,

    sampler=sampler,

    num_workers=NUM_WORKERS,

    pin_memory=True,

    persistent_workers=(
        NUM_WORKERS > 0
    )
)


# ================================================================
# 14. VALIDATION LOADERS
# ================================================================

val_pos = val_df[
    val_df.has_lesion == 1
]

val_neg = val_df[
    val_df.has_lesion == 0
]


fast_pos = val_pos.sample(

    n=min(
        FAST_VAL_POSITIVES,
        len(val_pos)
    ),

    random_state=SEED
)


fast_neg = val_neg.sample(

    n=min(
        FAST_VAL_NEGATIVES,
        len(val_neg)
    ),

    random_state=SEED
)


fast_val_df = pd.concat(
    [
        fast_pos,
        fast_neg
    ]
).reset_index(drop=True)


fast_val_loader = DataLoader(

    Dataset(
        dataframe_to_files(
            fast_val_df
        ),
        transform=eval_transforms
    ),

    batch_size=1,

    shuffle=False,

    num_workers=0,

    pin_memory=True
)


full_val_loader = DataLoader(

    Dataset(
        val_files,
        transform=eval_transforms
    ),

    batch_size=1,

    shuffle=False,

    num_workers=0,

    pin_memory=True
)


print(
    "\nSamples / epoch:",
    len(train_loader)
)

print(
    "Fast validation:",
    len(fast_val_df)
)

print(
    "Full validation:",
    len(val_df)
)


# ================================================================
# 15. MODEL
# ================================================================

model = SegResNet(

    spatial_dims=3,

    init_filters=16,

    in_channels=1,

    out_channels=2,

    dropout_prob=0.10,

    blocks_down=(
        1,
        2,
        2,
        4
    ),

    blocks_up=(
        1,
        1,
        1
    )

).to(DEVICE)


total_params = sum(
    p.numel()
    for p in model.parameters()
)


print(
    "\nModel parameters:",
    f"{total_params:,}"
)


# ================================================================
# 16. DOWNLOAD SuPreM
# ================================================================

download_file(
    SUPREM_URL,
    SUPREM_WEIGHTS
)

print(
    "SuPreM weights:",
    round(
        os.path.getsize(
            SUPREM_WEIGHTS
        ) / 1024**2,
        1
    ),
    "MB"
)


# ================================================================
# 17. FLEXIBLE SuPreM LOADER
# ================================================================

def load_suprem_weights(
    model,
    checkpoint_path
):

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )


    if isinstance(
        ckpt,
        dict
    ):

        if "net" in ckpt:

            source = ckpt["net"]

        elif "state_dict" in ckpt:

            source = ckpt[
                "state_dict"
            ]

        elif "model_state_dict" in ckpt:

            source = ckpt[
                "model_state_dict"
            ]

        else:

            source = ckpt

    else:

        source = ckpt


    target = model.state_dict()

    loaded = {}

    loaded_numel = 0


    for key, value in source.items():

        if not torch.is_tensor(
            value
        ):
            continue


        candidates = [
            key
        ]


        if key.startswith("module."):

            candidates.append(
                key[len("module."):]
            )


        if key.startswith("model."):

            candidates.append(
                key[len("model."):]
            )


        parts = key.split(".")

        if len(parts) > 1:

            candidates.append(
                ".".join(parts[1:])
            )


        candidates = list(
            dict.fromkeys(candidates)
        )


        for candidate in candidates:

            if (
                candidate in target
                and
                target[candidate].shape
                ==
                value.shape
            ):

                loaded[
                    candidate
                ] = value

                loaded_numel += (
                    value.numel()
                )

                break


    state = target.copy()

    state.update(
        loaded
    )

    model.load_state_dict(
        state
    )


    total_numel = sum(
        p.numel()
        for p in target.values()
    )


    fraction = (
        loaded_numel
        /
        total_numel
    )


    print("\nSUPREM INITIALIZATION")

    print(
        "Loaded tensors:",
        len(loaded),
        "/",
        len(target)
    )

    print(
        "Loaded parameters:",
        f"{100*fraction:.2f}%"
    )


    if fraction < 0.60:

        raise RuntimeError(
            "Less than 60% of SuPreM weights loaded. "
            "Stopping because model architecture "
            "probably does not match checkpoint."
        )


    return fraction


# ================================================================
# 18. LOSS FUNCTIONS
# ================================================================

def soft_dice_loss(
    probability,
    target,
    eps=1e-6
):

    dims = tuple(
        range(
            1,
            probability.ndim
        )
    )

    intersection = (
        probability
        *
        target
    ).sum(
        dim=dims
    )

    denominator = (
        probability.sum(
            dim=dims
        )
        +
        target.sum(
            dim=dims
        )
    )

    dice = (
        2 * intersection + eps
    ) / (
        denominator + eps
    )

    return (
        1.0 - dice
    )


def tversky_loss(
    probability,
    target,
    alpha=0.55,
    beta=0.45,
    eps=1e-6
):

    dims = tuple(
        range(
            1,
            probability.ndim
        )
    )

    tp = (
        probability
        *
        target
    ).sum(
        dim=dims
    )

    fp = (
        probability
        *
        (1.0 - target)
    ).sum(
        dim=dims
    )

    fn = (
        (1.0 - probability)
        *
        target
    ).sum(
        dim=dims
    )

    score = (
        tp + eps
    ) / (
        tp
        +
        alpha * fp
        +
        beta * fn
        +
        eps
    )

    return (
        1.0 - score
    )


def focal_bce(
    logits,
    target,
    gamma=2.0
):

    bce = (
        F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none"
        )
    )

    pt = torch.exp(
        -bce
    )

    focal = (
        (1.0 - pt) ** gamma
        *
        bce
    )

    dims = tuple(
        range(
            1,
            focal.ndim
        )
    )

    return focal.mean(
        dim=dims
    )


# ================================================================
# 19. COMBINED V5 LOSS
# ================================================================

def loss_function(
    logits,
    target,
    pancreas_valid,
    has_lesion
):

    prob = torch.sigmoid(
        logits
    )


    # ------------------------------------------------
    # channels
    # ------------------------------------------------

    pan_prob = prob[:, 0:1]

    les_prob = prob[:, 1:2]


    pan_target = target[:, 0:1]

    les_target = target[:, 1:2]


    pancreas_valid = (
        pancreas_valid
        .float()
        .view(-1)
    )

    has_lesion = (
        has_lesion
        .float()
        .view(-1)
    )


    # ------------------------------------------------
    # PANCREAS LOSS
    # ------------------------------------------------

    pan_dice = soft_dice_loss(
        pan_prob,
        pan_target
    )


    pan_bce = (

        F.binary_cross_entropy_with_logits(

            logits[:, 0:1],

            pan_target,

            reduction="none"

        )

        .mean(
            dim=(1, 2, 3, 4)
        )
    )


    pan_case_loss = (
        0.70 * pan_dice
        +
        0.30 * pan_bce
    )


    if pancreas_valid.sum() > 0:

        pancreas_loss = (

            pan_case_loss
            *
            pancreas_valid

        ).sum() / (
            pancreas_valid.sum()
            +
            1e-6
        )

    else:

        pancreas_loss = (
            logits.sum()
            *
            0.0
        )


    # ------------------------------------------------
    # LESION TVERSKY
    #
    # alpha > V4:
    # stronger false-positive penalty.
    # ------------------------------------------------

    lesion_tversky = (

        tversky_loss(

            les_prob,

            les_target,

            alpha=0.55,

            beta=0.45

        ).mean()
    )


    # ------------------------------------------------
    # LESION FOCAL BCE
    # ------------------------------------------------

    lesion_focal = (

        focal_bce(

            logits[:, 1:2],

            les_target,

            gamma=2.0

        ).mean()
    )


    # ------------------------------------------------
    # NEGATIVE PATIENT FALSE-POSITIVE PENALTY
    #
    # This is important for your V4 specificity issue.
    # ------------------------------------------------

    negative_mask = (
        1.0 - has_lesion
    )


    per_case_false_probability = (

        les_prob.mean(
            dim=(1, 2, 3, 4)
        )
    )


    if negative_mask.sum() > 0:

        negative_fp_loss = (

            per_case_false_probability
            *
            negative_mask

        ).sum() / (
            negative_mask.sum()
            +
            1e-6
        )

    else:

        negative_fp_loss = (
            logits.sum()
            *
            0.0
        )


    # ------------------------------------------------
    # ANATOMICAL CONSISTENCY
    #
    # Lesion probability should normally not exceed
    # pancreas probability substantially.
    # ------------------------------------------------

    anatomy_loss = (

        torch.relu(
            les_prob
            -
            pan_prob
        )

        .mean()
    )


    # ------------------------------------------------
    # TOTAL
    # ------------------------------------------------

    total = (

        0.60 * pancreas_loss

        +

        2.20 * lesion_tversky

        +

        0.60 * lesion_focal

        +

        0.60 * negative_fp_loss

        +

        0.15 * anatomy_loss
    )


    return total


# ================================================================
# 20. OPTIMIZER
# ================================================================

head_parameters = []

backbone_parameters = []


for name, param in model.named_parameters():

    if "conv_final" in name:

        head_parameters.append(
            param
        )

    else:

        backbone_parameters.append(
            param
        )


optimizer = torch.optim.AdamW(

    [

        {
            "params":
                backbone_parameters,

            "lr":
                BASE_LR,
        },

        {
            "params":
                head_parameters,

            "lr":
                HEAD_LR,
        },

    ],

    weight_decay=WEIGHT_DECAY
)


scheduler = (

    torch.optim.lr_scheduler
    .CosineAnnealingLR(

        optimizer,

        T_max=MAX_EPOCHS,

        eta_min=1e-6
    )
)


scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_AMP
)


# ================================================================
# 21. METRIC HELPERS
# ================================================================

def binary_dice(
    pred,
    gt
):

    pred = pred.astype(bool)

    gt = gt.astype(bool)


    denominator = (
        pred.sum()
        +
        gt.sum()
    )


    if denominator == 0:

        return np.nan


    return float(

        2
        *
        np.logical_and(
            pred,
            gt
        ).sum()

        /

        denominator
    )


def remove_small_components(
    mask,
    min_voxels
):

    mask = mask.astype(bool)


    labeled, count = ndimage.label(
        mask
    )


    if count == 0:

        return mask


    sizes = np.bincount(
        labeled.ravel()
    )


    output = np.zeros_like(
        mask,
        dtype=bool
    )


    for component_id in range(
        1,
        count + 1
    ):

        if sizes[component_id] >= min_voxels:

            output[
                labeled == component_id
            ] = True


    return output


def tumor_detection_counts(
    pred,
    gt
):

    """
    A GT tumor is counted detected if any predicted
    lesion component overlaps it.
    """

    gt_labeled, gt_n = ndimage.label(
        gt.astype(bool)
    )


    detected = 0


    for component in range(
        1,
        gt_n + 1
    ):

        gt_component = (
            gt_labeled
            ==
            component
        )


        if np.logical_and(
            pred,
            gt_component
        ).any():

            detected += 1


    return detected, gt_n


def patient_continuous_score(
    lesion_prob,
    pan_prob
):

    # Very low pancreas threshold provides a robust
    # candidate region without allowing whole-body
    # false-positive score dominance.

    roi = (
        pan_prob > 0.10
    )


    if roi.sum() >= 100:

        scores = lesion_prob[
            roi
        ]

    else:

        scores = lesion_prob.ravel()


    # More robust than raw max.
    return float(
        np.quantile(
            scores,
            0.999
        )
    )


# ================================================================
# 22. INFERENCE CACHE
#
# Run network once per validation volume.
# Threshold / component tuning happens later without
# repeating GPU inference.
# ================================================================

def generate_validation_predictions(
    model,
    loader,
    description
):

    model.eval()

    records = []


    with torch.inference_mode():

        for batch in tqdm(
            loader,
            desc=description
        ):

            image = (
                batch["image"]
                .to(
                    DEVICE,
                    non_blocking=True
                )
            )


            label = (
                batch["label"]
                .float()
                .to(
                    DEVICE,
                    non_blocking=True
                )
            )


            with torch.amp.autocast(

                device_type="cuda",

                enabled=USE_AMP
            ):

                logits = sliding_window_inference(

                    image,

                    roi_size=PATCH_SIZE,

                    sw_batch_size=1,

                    predictor=model,

                    overlap=0.50,

                    mode="gaussian"
                )


            probabilities = (

                torch.sigmoid(
                    logits
                )[0]

                .float()

                .cpu()

                .numpy()
            )


            gt = (

                label[0]

                .float()

                .cpu()

                .numpy()
            )


            case_value = batch[
                "case_id"
            ]


            if isinstance(
                case_value,
                (list, tuple)
            ):

                case_id = str(
                    case_value[0]
                )

            else:

                case_id = str(
                    case_value
                )


            records.append({

                "case_id":
                    case_id,

                "pan_prob":
                    probabilities[0],

                "lesion_prob":
                    probabilities[1],

                "gt_pan":
                    gt[0] > 0.5,

                "gt_lesion":
                    gt[1] > 0.5,
            })


            del (
                image,
                label,
                logits,
                probabilities
            )


            torch.cuda.empty_cache()


    return records


# ================================================================
# 23. OPERATING POINT EVALUATION
# ================================================================

def evaluate_operating_point(

    records,

    lesion_threshold,

    pancreas_gate,

    min_component_voxels,

    dilation_iterations=3
):

    dices = []

    pancreas_dices = []

    patient_gt = []

    patient_pred = []

    patient_scores = []

    total_tumors = 0

    detected_tumors = 0


    for item in records:

        pan_prob = item[
            "pan_prob"
        ]

        lesion_prob = item[
            "lesion_prob"
        ]

        gt_pan = item[
            "gt_pan"
        ]

        gt_lesion = item[
            "gt_lesion"
        ]


        # ----------------------------------------
        # pancreas
        # ----------------------------------------

        pred_pan = (
            pan_prob > 0.50
        )


        pdice = binary_dice(
            pred_pan,
            gt_pan
        )


        if not np.isnan(pdice):

            pancreas_dices.append(
                pdice
            )


        # ----------------------------------------
        # slightly dilated predicted pancreas
        # ----------------------------------------

        gate_mask = (
            pan_prob
            >
            pancreas_gate
        )


        if dilation_iterations > 0:

            gate_mask = (
                ndimage.binary_dilation(

                    gate_mask,

                    iterations=dilation_iterations
                )
            )


        # ----------------------------------------
        # lesion
        # ----------------------------------------

        predicted_lesion = (

            (lesion_prob > lesion_threshold)

            &

            gate_mask
        )


        predicted_lesion = (
            remove_small_components(

                predicted_lesion,

                min_component_voxels
            )
        )


        # ----------------------------------------
        # case-level status
        # ----------------------------------------

        gt_positive = bool(
            gt_lesion.any()
        )

        pred_positive = bool(
            predicted_lesion.any()
        )


        patient_gt.append(
            int(gt_positive)
        )

        patient_pred.append(
            int(pred_positive)
        )


        patient_scores.append(

            patient_continuous_score(

                lesion_prob,

                pan_prob
            )
        )


        # ----------------------------------------
        # lesion Dice
        # ----------------------------------------

        if gt_positive:

            d = binary_dice(

                predicted_lesion,

                gt_lesion
            )


            if not np.isnan(d):

                dices.append(
                    d
                )


        # ----------------------------------------
        # tumor-wise sensitivity
        # ----------------------------------------

        det, total = (
            tumor_detection_counts(

                predicted_lesion,

                gt_lesion
            )
        )


        detected_tumors += det

        total_tumors += total


    patient_gt = np.asarray(
        patient_gt
    )

    patient_pred = np.asarray(
        patient_pred
    )


    tp = int(
        (
            (patient_gt == 1)
            &
            (patient_pred == 1)
        ).sum()
    )

    fn = int(
        (
            (patient_gt == 1)
            &
            (patient_pred == 0)
        ).sum()
    )

    tn = int(
        (
            (patient_gt == 0)
            &
            (patient_pred == 0)
        ).sum()
    )

    fp = int(
        (
            (patient_gt == 0)
            &
            (patient_pred == 1)
        ).sum()
    )


    p_sen = (
        tp
        /
        max(
            tp + fn,
            1
        )
    )


    specificity = (
        tn
        /
        max(
            tn + fp,
            1
        )
    )


    t_sen = (
        detected_tumors
        /
        max(
            total_tumors,
            1
        )
    )


    mean_dice = (
        float(
            np.mean(dices)
        )
        if len(dices)
        else 0.0
    )


    pancreas_dice = (
        float(
            np.mean(
                pancreas_dices
            )
        )
        if len(pancreas_dices)
        else 0.0
    )


    try:

        auc = float(

            roc_auc_score(

                patient_gt,

                patient_scores
            )
        )

    except Exception:

        auc = 0.0


    # ============================================================
    # Validation selection score
    #
    # Specificity receives substantial weight because it was the
    # largest weakness in V4.
    #
    # This is ONLY for selecting checkpoint / operating parameters.
    # It does not replace official PanTS metrics.
    # ============================================================

    selection_score = (

        0.30 * mean_dice

        +

        0.20 * p_sen

        +

        0.20 * t_sen

        +

        0.25 * specificity

        +

        0.05 * auc
    )


    return {

        "lesion_threshold":
            float(lesion_threshold),

        "pancreas_gate":
            float(pancreas_gate),

        "min_component_voxels":
            int(min_component_voxels),

        "dilation_iterations":
            int(dilation_iterations),

        "pancreas_dice":
            pancreas_dice,

        "lesion_dice":
            mean_dice,

        "p_sen":
            float(p_sen),

        "t_sen":
            float(t_sen),

        "specificity":
            float(specificity),

        "auc":
            float(auc),

        "TP":
            tp,

        "TN":
            tn,

        "FP":
            fp,

        "FN":
            fn,

        "detected_tumors":
            int(detected_tumors),

        "total_tumors":
            int(total_tumors),

        "selection_score":
            float(selection_score),
    }


# ================================================================
# 24. VALIDATION PARAMETER SEARCH
# ================================================================

LESION_THRESHOLDS = [
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]


PANCREAS_GATES = [
    0.05,
    0.10,
    0.15,
    0.20,
]


MIN_COMPONENT_VOXELS = [
    10,
    20,
    40,
    80,
    120,
    200,
]


DILATIONS = [
    2,
    3,
    4,
]


def tune_operating_point(
    records
):

    results = []


    for lesion_threshold in LESION_THRESHOLDS:

        for pancreas_gate in PANCREAS_GATES:

            for min_component in MIN_COMPONENT_VOXELS:

                for dilation in DILATIONS:

                    metrics = (
                        evaluate_operating_point(

                            records,

                            lesion_threshold,

                            pancreas_gate,

                            min_component,

                            dilation
                        )
                    )


                    results.append(
                        metrics
                    )


    results_df = pd.DataFrame(
        results
    )


    # ------------------------------------------------------------
    # Prefer clinically sensible operating points:
    # avoid choosing a threshold giving high Dice but poor
    # specificity or catastrophic sensitivity.
    # ------------------------------------------------------------

    candidates = results_df[

        (
            results_df["specificity"]
            >=
            0.80
        )

        &

        (
            results_df["p_sen"]
            >=
            0.60
        )

    ]


    if len(candidates) == 0:

        candidates = results_df


    best_row = (

        candidates

        .sort_values(
            "selection_score",
            ascending=False
        )

        .iloc[0]
    )


    return (
        best_row.to_dict(),
        results_df
    )


# ================================================================
# 25. CHECKPOINT / RESUME
# ================================================================

history = []

start_epoch = 1

best_selection_score = -1.0

epochs_without_improvement = 0


if os.path.exists(
    HISTORY_FILE
):

    try:

        history = (

            pd.read_csv(
                HISTORY_FILE
            )

            .to_dict(
                "records"
            )
        )

    except Exception:

        history = []


if os.path.exists(
    LAST_MODEL
):

    print(
        "\nExisting V5 checkpoint found."
    )

    print(
        "RESUMING TRAINING..."
    )


    checkpoint = torch.load(

        LAST_MODEL,

        map_location=DEVICE,

        weights_only=False
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )


    scheduler.load_state_dict(
        checkpoint[
            "scheduler_state_dict"
        ]
    )


    if (
        "scaler_state_dict"
        in checkpoint
    ):

        scaler.load_state_dict(
            checkpoint[
                "scaler_state_dict"
            ]
        )


    start_epoch = (

        int(
            checkpoint["epoch"]
        )

        +
        1
    )


    best_selection_score = float(

        checkpoint.get(

            "best_selection_score",

            -1.0
        )
    )


    epochs_without_improvement = int(

        checkpoint.get(

            "epochs_without_improvement",

            0
        )
    )


    print(
        "Resuming at epoch:",
        start_epoch
    )


else:

    load_suprem_weights(

        model,

        SUPREM_WEIGHTS
    )


# ================================================================
# 26. GPU SMOKE TEST
# ================================================================

print("\nRunning GPU smoke test...")


model.eval()


with torch.inference_mode():

    dummy = torch.randn(

        1,

        1,

        *PATCH_SIZE,

        device=DEVICE
    )


    with torch.amp.autocast(

        "cuda",

        enabled=USE_AMP
    ):

        dummy_out = model(
            dummy
        )


print(
    "Input :",
    tuple(dummy.shape)
)

print(
    "Output:",
    tuple(dummy_out.shape)
)


del dummy
del dummy_out

torch.cuda.empty_cache()


# ================================================================
# 27. TRAINING
# ================================================================

print("\n" + "=" * 70)
print("STARTING PanTS V5 TRAINING")
print("=" * 70)


for epoch in range(
    start_epoch,
    MAX_EPOCHS + 1
):

    epoch_start = time.time()

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    running_loss = 0.0


    progress = tqdm(

        enumerate(
            train_loader
        ),

        total=len(
            train_loader
        ),

        desc=(
            f"Epoch "
            f"{epoch}/{MAX_EPOCHS}"
        )
    )


    for step, batch in progress:

        image = (
            batch["image"]
            .to(
                DEVICE,
                non_blocking=True
            )
        )


        label = (
            batch["label"]
            .float()
            .to(
                DEVICE,
                non_blocking=True
            )
        )


        pancreas_valid = (

            torch.as_tensor(

                batch[
                    "pancreas_valid"
                ]
            )

            .to(
                DEVICE
            )
        )


        has_lesion = (

            torch.as_tensor(

                batch[
                    "has_lesion"
                ]
            )

            .to(
                DEVICE
            )
        )


        with torch.amp.autocast(

            device_type="cuda",

            enabled=USE_AMP
        ):

            logits = model(
                image
            )


            loss = loss_function(

                logits,

                label,

                pancreas_valid,

                has_lesion
            )


            loss_for_backward = (

                loss
                /
                GRAD_ACCUM
            )


        scaler.scale(
            loss_for_backward
        ).backward()


        if (

            (
                step + 1
            )
            %
            GRAD_ACCUM
            ==
            0

            or

            (
                step + 1
            )
            ==
            len(train_loader)

        ):

            scaler.unscale_(
                optimizer
            )


            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                max_norm=5.0
            )


            scaler.step(
                optimizer
            )


            scaler.update()


            optimizer.zero_grad(
                set_to_none=True
            )


        running_loss += float(
            loss.item()
        )


        avg_loss = (

            running_loss

            /
            (
                step + 1
            )
        )


        progress.set_postfix({

            "loss":
                f"{avg_loss:.4f}",

            "lr":
                f"{optimizer.param_groups[0]['lr']:.2e}"
        })


        del (
            image,
            label,
            logits,
            loss
        )


    scheduler.step()


    train_loss = (

        running_loss
        /
        max(
            len(train_loader),
            1
        )
    )


    # ============================================================
    # VALIDATION
    # ============================================================

    do_full_validation = (

        epoch % FULL_VAL_EVERY == 0

        or

        epoch == 1

        or

        epoch == MAX_EPOCHS
    )


    if do_full_validation:

        validation_loader = (
            full_val_loader
        )

        validation_name = (
            "Full validation"
        )

    else:

        validation_loader = (
            fast_val_loader
        )

        validation_name = (
            "Fast validation"
        )


    print(
        f"\n{validation_name}..."
    )


    validation_records = (
        generate_validation_predictions(

            model,

            validation_loader,

            validation_name
        )
    )


    best_config, tuning_df = (
        tune_operating_point(

            validation_records
        )
    )


    tuning_df.to_csv(

        f"{RESULT_DIR}/"

        f"threshold_search_"
        f"epoch_{epoch:03d}.csv",

        index=False
    )


    score = float(

        best_config[
            "selection_score"
        ]
    )


    print("\n" + "-" * 70)

    print(
        "Epoch:",
        epoch
    )

    print(
        "Train loss:",
        round(
            train_loss,
            5
        )
    )

    print(
        "Lesion DSC:",
        round(
            best_config[
                "lesion_dice"
            ],
            4
        )
    )

    print(
        "P-Sen:",
        round(
            best_config[
                "p_sen"
            ],
            4
        )
    )

    print(
        "T-Sen:",
        round(
            best_config[
                "t_sen"
            ],
            4
        )
    )

    print(
        "Specificity:",
        round(
            best_config[
                "specificity"
            ],
            4
        )
    )

    print(
        "AUC:",
        round(
            best_config[
                "auc"
            ],
            4
        )
    )

    print(
        "Threshold:",
        best_config[
            "lesion_threshold"
        ]
    )

    print(
        "Pancreas gate:",
        best_config[
            "pancreas_gate"
        ]
    )

    print(
        "Min component:",
        int(
            best_config[
                "min_component_voxels"
            ]
        )
    )

    print(
        "Selection score:",
        round(
            score,
            4
        )
    )

    print("-" * 70)


    # ============================================================
    # HISTORY
    # ============================================================

    epoch_record = {

        "epoch":
            epoch,

        "train_loss":
            train_loss,

        "full_validation":
            int(
                do_full_validation
            ),

        **best_config,

        "minutes":
            (
                time.time()
                -
                epoch_start
            )
            /
            60.0
    }


    history.append(
        epoch_record
    )


    pd.DataFrame(
        history
    ).to_csv(
        HISTORY_FILE,
        index=False
    )


    # ============================================================
    # SAVE LAST CHECKPOINT EVERY EPOCH
    # ============================================================

    checkpoint = {

        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "scaler_state_dict":
            scaler.state_dict(),

        "best_selection_score":
            best_selection_score,

        "epochs_without_improvement":
            epochs_without_improvement,

        "validation_config":
            best_config,
    }


    torch.save(
        checkpoint,
        LAST_MODEL
    )


    # ============================================================
    # BEST CHECKPOINT:
    #
    # Only allow full validation epochs to become final "best".
    # Prevent tiny fast-validation set from determining final model.
    # ============================================================

    if (
        do_full_validation
        and
        score > best_selection_score
    ):

        best_selection_score = score

        epochs_without_improvement = 0


        checkpoint[
            "best_selection_score"
        ] = best_selection_score


        torch.save(
            checkpoint,
            BEST_MODEL
        )


        with open(

            BEST_CONFIG_FILE,

            "w"

        ) as f:

            json.dump(

                best_config,

                f,

                indent=2
            )


        print(
            "\n*** NEW BEST V5 MODEL SAVED ***"
        )


    elif do_full_validation:

        epochs_without_improvement += 1


    del validation_records

    torch.cuda.empty_cache()


    # ============================================================
    # EARLY STOP
    # ============================================================

    if (

        epochs_without_improvement
        >=
        EARLY_STOP_PATIENCE

    ):

        print(
            "\nEarly stopping."
        )

        break


print("\n" + "=" * 70)
print("TRAINING FINISHED")
print("=" * 70)

print(
    "\nBest checkpoint:"
)

print(
    BEST_MODEL
)

print(
    "\nTraining history:"
)

print(
    HISTORY_FILE
)

print(
    "\nBest validation configuration:"
)

print(
    BEST_CONFIG_FILE
)

import os
import glob

print("MyDrive folders:")
print(os.listdir("/content/drive/MyDrive")[:100])

print("\nSearching for PanTS CT files...")
ct_files = glob.glob(
    "/content/drive/MyDrive/**/PanTS_*/ct.nii.gz",
    recursive=True
)

print("CT files found:", len(ct_files))

for x in ct_files[:20]:
    print(x)

print("\nSearching for PanTS labels...")
label_files = glob.glob(
    "/content/drive/MyDrive/**/PanTS_*/combined_labels.nii.gz",
    recursive=True
)

print("Labels found:", len(label_files))

for x in label_files[:20]:
    print(x)

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os

print(os.listdir("/content/drive"))
print(os.path.exists("/content/drive/MyDrive"))

print(os.listdir("/content/drive/MyDrive")[:50])

import glob

ct_files = glob.glob(
    "/content/drive/MyDrive/**/PanTS_*/ct.nii.gz",
    recursive=True
)

label_files = glob.glob(
    "/content/drive/MyDrive/**/PanTS_*/combined_labels.nii.gz",
    recursive=True
)

print("CT files:", len(ct_files))
print("Labels:", len(label_files))

print("\nFirst CT:")
print(ct_files[:3])

print("\nFirst labels:")
print(label_files[:3])

# ================================================================
# PanTS DATA SETUP FOR COLAB
# Data lives in temporary Colab storage.
# Checkpoints/results stay permanently on Google Drive.
# ================================================================

import os
import glob
import subprocess

DRIVE_ROOT = "/content/drive/MyDrive"

PROJECT = f"{DRIVE_ROOT}/PanTS_Project/V5"

CHECKPOINT_DIR = f"{PROJECT}/checkpoints"
RESULT_DIR = f"{PROJECT}/results"
MANIFEST_DIR = f"{PROJECT}/manifests"

for d in [
    PROJECT,
    CHECKPOINT_DIR,
    RESULT_DIR,
    MANIFEST_DIR,
]:
    os.makedirs(d, exist_ok=True)


# ------------------------------------------------
# Temporary Colab dataset location
# ------------------------------------------------

LOCAL_DATA = "/content/PanTS/data"

IMAGE_ROOT = f"{LOCAL_DATA}/ImageTr"
LABEL_ROOT = f"{LOCAL_DATA}/LabelTr"

os.makedirs(IMAGE_ROOT, exist_ok=True)
os.makedirs(LABEL_ROOT, exist_ok=True)


# ------------------------------------------------
# PanTSMini — first 1000 cases
# ------------------------------------------------

IMAGE_ARCHIVE = (
    f"{LOCAL_DATA}/PanTSMini_ImageTr_00000001_00001000.tar.gz"
)

LABEL_ARCHIVE = (
    f"{LOCAL_DATA}/PanTSMini_Label.tar.gz"
)

IMAGE_URL = (
    "https://huggingface.co/datasets/"
    "BodyMaps/PanTSMini/resolve/main/"
    "PanTSMini_ImageTr_00000001_00001000.tar.gz"
    "?download=true"
)

LABEL_URL = (
    "https://www.cs.jhu.edu/~zongwei/dataset/"
    "PanTSMini_Label.tar.gz"
)


def run(cmd):
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def download(url, path):

    if os.path.exists(path):
        print("Already downloaded:", path)
        return

    run([
        "wget",
        "-c",
        "-O",
        path,
        url
    ])


# ------------------------------------------------
# Download CT
# ------------------------------------------------

ct_count = len(
    glob.glob(
        f"{IMAGE_ROOT}/PanTS_*/ct.nii.gz"
    )
)

print("Existing CT cases:", ct_count)


if ct_count < 1000:

    print("\nDownloading PanTS CT data...")

    download(
        IMAGE_URL,
        IMAGE_ARCHIVE
    )

    print("\nExtracting CT data...")

    run([
        "tar",
        "-xzf",
        IMAGE_ARCHIVE,
        "-C",
        IMAGE_ROOT
    ])


# ------------------------------------------------
# Download labels
# ------------------------------------------------

label_count = len(
    glob.glob(
        f"{LABEL_ROOT}/PanTS_*/combined_labels.nii.gz"
    )
)

print("Existing labels:", label_count)


if label_count < 1000:

    print("\nDownloading PanTS labels...")

    download(
        LABEL_URL,
        LABEL_ARCHIVE
    )

    print("\nExtracting labels...")

    run([
        "tar",
        "-xzf",
        LABEL_ARCHIVE,
        "-C",
        LABEL_ROOT
    ])


# ------------------------------------------------
# Verify
# ------------------------------------------------

ct_files = glob.glob(
    f"{IMAGE_ROOT}/PanTS_*/ct.nii.gz"
)

label_files = glob.glob(
    f"{LABEL_ROOT}/PanTS_*/combined_labels.nii.gz"
)

print("\n" + "=" * 60)
print("DATA CHECK")
print("=" * 60)

print("CT files:", len(ct_files))
print("Label files:", len(label_files))

print("\nFirst CT:")
print(ct_files[:3])

print("\nFirst label:")
print(label_files[:3])


if len(ct_files) == 0:
    raise RuntimeError("CT extraction failed.")

if len(label_files) == 0:
    raise RuntimeError("Label extraction failed.")
