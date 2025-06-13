#!/usr/bin/env bash

%runscript
  exec "$@"

cd /base_path/PiPViT/mains
echo "Where am I ..."
ls
nvidia-smi

#train script
python Smain.py --config_path /base_path/PiPViT/config/Train/train/OCTDrusen/Sconfig.yaml


echo "I am done ..."