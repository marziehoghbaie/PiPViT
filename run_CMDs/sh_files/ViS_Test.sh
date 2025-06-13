#!/usr/bin/env bash

%runscript
  exec "$@"

cd /base_path/PiPViT/mains
echo "Where am I ..."
ls
nvidia-smi

python Smain_vis.py  --config_path /base_path/PiPViT/config/Vis/OCT5K/Sconfig.yaml

echo "I am done ..."