#!/usr/bin/env bash 
set -eu  
epochs=200
# constrainted by GPU number & memory
batch_size=32
gpuid=0
num_workers=32
cpt_dir=/node/hzl/expriment/SEF_PNet_icassp2025_github/demo
#resume=
#[ $# -ne 1 ] && echo "Script error: $0 <gpuid> <cpt-id>" && exit 1
./nnet/train_unet_tse_steplr_clip.py \
  --gpu $gpuid \
  --epochs $epochs \
  --batch-size $batch_size \
  --num-workers $num_workers \
  --checkpoint $cpt_dir \
> train.log 2>&1
