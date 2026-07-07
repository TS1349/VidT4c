#!/usr/bin/env bash
# Download pretrained backbone weights into ./pretrained/.
# Weights are hosted on Google Drive (too large for git). Files:
#   vemt_v0.pth (360M)        VideoMAE  (ours)
#   cbramod.pth (19M)         CBraMod   (ours, EEG)
#   checkpoint-199.pth (1.17G) AdaMAE   (benchmark)
#   tsf.pth (465M)            TSF        (benchmark)
#   labram.pth (93M)          LaBraM    (benchmark, EEG)
# ViViT / Swin / REVE self-download from HuggingFace/torchvision (no file needed).
set -e
cd "$(dirname "$0")"
mkdir -p pretrained

# Shared Google Drive folder (anyone-with-link) holding the 5 backbone weights.
GDRIVE_FOLDER_URL="https://drive.google.com/drive/folders/15Vxe74U2BXHoWDuazu6tp5_I284rxSet"

if ! command -v gdown >/dev/null 2>&1; then
  echo "installing gdown..."; pip install gdown
fi

echo "downloading pretrained weights into ./pretrained/ ..."
gdown --folder "$GDRIVE_FOLDER_URL" -O pretrained --remaining-ok

echo "done. verifying:"
for f in vemt_v0.pth cbramod.pth checkpoint-199.pth tsf.pth labram.pth; do
  if [ -f "pretrained/$f" ]; then echo "  OK  $f"; else echo "  MISSING  $f"; fi
done
