#!/bin/bash 
set -eu

checkpoint=/node/hzl/expriment/libri2mix_min_wav8k/SEF_PNet
gpuid=0

data_root=/node/hzl/data/data_libri2mix_s1_min_wav8k/test

mix_scp=$data_root/mix_clean.scp 
spk1_scp=$data_root/s1.scp 
aux_scp=$data_root/auxs1.scp 

cal_sdr=1

./nnet/evaluate.py \
  --checkpoint $checkpoint \
  --gpuid $gpuid \
  --mix_scp $mix_scp \
  --ref_scp $spk1_scp \
  --aux_scp $aux_scp \
  --cal_sdr $cal_sdr \
> eval.log 2>&1

echo "eval done!"
