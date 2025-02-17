#!/bin/bash 
set -eu
checkpoint=/node/hzl/expriment/libri2mix_min_wav8k/SEF_PNet
gpuid=0
data_root=/node/hzl/data/data_libri2mix_s1_min_wav8k/test

mix_scp=$data_root/mix_clean.scp
ref_scp=$data_root/s1.scp
aux_scp=$data_root/auxs1.scp 

fs=8000
dump_dir=/node/hzl/data/enhanced_speech

./nnet/separate.py \
  --checkpoint $checkpoint \
  --gpuid $gpuid \
  --mix_scp $mix_scp \
  --ref_scp $ref_scp \
  --aux_scp $aux_scp \
  --fs $fs \
  --dump-dir $dump_dir \
 > separate.log 2>&1

echo "Separate done!"
