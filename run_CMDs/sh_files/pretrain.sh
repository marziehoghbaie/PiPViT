#!/usr/bin/env bash

%runscript
  exec "$@"

cd /base_path/PiPViT/mains
echo "Where am I ..."
ls
nvidia-smi

# single scale pretraining
python Smain.py --config_path /base_path/PiPViT/config/Pretrain/Pretrain_224/OCTDrusen/Sconfig_patch16_224.yaml
#multi-scale pretraining
python Smain_multi_scale_pretrain.py --config_path /base_path/PiPViT/config/Pretrain/Pretrain_224/OCTDrusen/Sconfig_patch16_224.yaml


echo "I am done ..."