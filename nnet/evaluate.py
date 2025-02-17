#!/usr/bin/env python3

import os
import time
import argparse
import torch as th
import numpy as np
from mir_eval.separation import bss_eval_sources
from pesq import pesq as pesq2
from pypesq import pesq as pesq1
from pystoi.stoi import stoi
from SEF_PNet_pse import SEF_PNet
from libs.utils import load_json, get_logger
from libs.dataset_tse import Dataset

def evaluate(args, model_file, logger):
    start = time.time()
    total_SISNR = 0
    total_SISNRi = 0
    total_PESQ = 0
    total_PESQi = 0
    total_PESQ2 = 0
    total_PESQi2 = 0
    total_STOI = 0
    total_STOIi = 0
    total_SDR = 0
    total_cnt = 0

    # Load model
    nnet_conf = load_json(args.checkpoint, "mdl.json")
    nnet = SEF_PNet(**nnet_conf)
    cpt_fname = os.path.join(args.checkpoint, model_file)
    cpt = th.load(cpt_fname, map_location="cpu")
    nnet.load_state_dict(cpt["model_state_dict"])
    logger.info("Loaded checkpoint from {}, epoch {:d}".format(
        cpt_fname, cpt["epoch"]))
    
    device = th.device(
        "cuda:{}".format(args.gpuid)) if args.gpuid >= 0 else th.device("cpu")
    nnet = nnet.to(device) if args.gpuid >= 0 else nnet
    nnet.eval()

    # Load data
    dataset = Dataset(mix_scp=args.mix_scp, ref_scp=args.ref_scp, aux_scp=args.aux_scp, sample_rate=8000)
    
    with th.no_grad():
        for i, data in enumerate(dataset):
            mix = th.tensor(data['mix'], dtype=th.float32, device=device)
            aux = th.tensor(data['aux'], dtype=th.float32, device=device)
                
            if args.gpuid >= 0:
                mix = mix.unsqueeze(0).to(device)
                aux = aux.unsqueeze(0).to(device)
                
            # Forward
            ref = data['ref']
            key = data['key']
            ests = nnet(mix, aux)
            ests = ests.cpu().numpy()
            mix = mix.squeeze(0).cpu().numpy()
            if ests.size != ref.size:
                end = min(ests.size, ref.size)
                ests = ests[:end]
                ref = ref[:end]
                mix = mix[:end]
                        
            # Compute metrics
            if args.cal_sdr == 1:
                SDR, sir, sar, popt = bss_eval_sources(ref, ests)
                total_SDR += SDR[0]
            SISNR, delta = cal_SISNRi(ests, ref, mix)
            PESQ, PESQi, PESQ2, PESQi2 = cal_PESQi(ests, ref, mix)
            STOI, STOIi = cal_STOIi(ests, ref, mix)
            if args.cal_sdr == 1:
                logger.info("Utt={:d} | SDR={:.2f} | SI-SNR={:.2f} | SI-SNRi={:.2f} | PESQ={:.2f} | PESQi={:.2f}| PESQ2={:.2f} | PESQi2={:.2f} | | STOI={:.2f} | STOIi={:.2f}".format(
                    total_cnt+1, SDR[0], SISNR, delta, PESQ, PESQi, PESQ2, PESQi2, STOI, STOIi))
            else:
                logger.info("Utt={:d} | SI-SNR={:.2f} | SI-SNRi={:.2f} | PESQ={:.2f} | PESQi={:.2f} | PESQ2={:.2f} | PESQi2={:.2f} | STOI={:.2f} | STOIi={:.2f}".format(
                    total_cnt+1, SISNR, delta, PESQ, PESQi, PESQ2, PESQi2, STOI, STOIi))
            total_SISNR += SISNR
            total_SISNRi += delta
            total_PESQ += PESQ
            total_PESQi += PESQi
            total_PESQ2 += PESQ2
            total_PESQi2 += PESQi2
            total_STOI += STOI
            total_STOIi += STOIi
            total_cnt += 1
    end = time.time()
    
    logger.info('Time Elapsed: {:.1f}s'.format(end-start))
    if args.cal_sdr == 1:
        logger.info("Average SDR: {0:.2f}".format(total_SDR / total_cnt))
    logger.info("Average SI-SNR: {:.2f}".format(total_SISNR / total_cnt))
    logger.info("Average SI-SNRi: {:.2f}".format(total_SISNRi / total_cnt))
    logger.info("Average PESQ: {:.2f}".format(total_PESQ / total_cnt))
    logger.info("Average PESQi: {:.2f}".format(total_PESQi / total_cnt))
    logger.info("Average PESQ2: {:.2f}".format(total_PESQ2 / total_cnt))
    logger.info("Average PESQi2: {:.2f}".format(total_PESQi2 / total_cnt))
    logger.info("Average STOI: {:.2f}".format(total_STOI / total_cnt))
    logger.info("Average STOIi: {:.2f}".format(total_STOIi / total_cnt))

def cal_SISNR(est, ref, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        est: separated signal, numpy.ndarray, [T]
        ref: reference signal, numpy.ndarray, [T]
    Returns:
        SISNR
    """ 
    assert len(est) == len(ref)
    est_zm = est - np.mean(est)
    ref_zm = ref - np.mean(ref)

    t = np.sum(est_zm * ref_zm) * ref_zm / (np.linalg.norm(ref_zm)**2 + eps)
        
    return 20 * np.log10(eps + np.linalg.norm(t) / (np.linalg.norm(est_zm - t) + eps))

def cal_SISNRi(est, ref, mix, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        est: separated signal, numpy.ndarray, [T]
        ref: reference signal, numpy.ndarray, [T]
    Returns:
        SISNR
    """ 
    assert len(est) == len(ref) == len(mix)
    sisnr1 = cal_SISNR(est, ref)
    sisnr2 = cal_SISNR(mix, ref)
    
    return sisnr1, sisnr1 - sisnr2
                         
def cal_PESQ(est, ref):
    assert len(est) == len(ref)
    mode ='nb'
    p = pesq1(ref, est,8000)
    p2 = pesq2(8000, ref, est, mode)
    return p,p2

def cal_PESQi(est, ref, mix):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        est: separated signal, numpy.ndarray, [T]
        ref: reference signal, numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(est) == len(ref) == len(mix)
    pesq1,pesq12 = cal_PESQ(est, ref)
    pesq2,pesq22= cal_PESQ(mix, ref)

    return pesq1, pesq1 - pesq2,pesq12,pesq12-pesq22

def cal_STOI(est, ref):
    assert len(est) == len(ref)
    p = stoi(ref, est, 8000)
    return p

def cal_STOIi(est, ref, mix):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        est: separated signal, numpy.ndarray, [T]
        ref: reference signal, numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(est) == len(ref) == len(mix)
    stoi1 = cal_STOI(est, ref)*100
    stoi2 = cal_STOI(mix, ref)*100

    return stoi1, stoi1 - stoi2

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate separation performance using Conv-TasNet')
    parser.add_argument('--checkpoint', type=str,
                        default='/node/hzl/expriment/SEF_PNet_icassp2025_github/demo',
                        help='Path to model directory containing checkpoints')
    parser.add_argument('--gpuid', type=int, default=0,
                        help="GPU device to offload model to, -1 means running on CPU")  
    parser.add_argument('--mix_scp', type=str,
                        default='/node/hzl/expriment/SEF_PNet_icassp2025_github/data/test/mix_clean.scp',
                        help='mix scp')
    parser.add_argument('--ref_scp', type=str,
                        default='/node/hzl/expriment/SEF_PNet_icassp2025_github/data/test/ref.scp',
                        help='ref scp')
    parser.add_argument('--aux_scp', type=str,
                        default='/node/hzl/expriment/SEF_PNet_icassp2025_github/data/test/auxs1.scp',
                        help='aux scp')    
    parser.add_argument('--cal_sdr', type=int, default=None,
                        help='Whether calculate SDR, add this option because calculation of SDR is very slow')

    args = parser.parse_args()

    
    # eval best.pt.tar
    best_model_file = "best.pt.tar"
    best_log_file = os.path.join(args.checkpoint, "eval_best.log")
    best_logger = get_logger(best_log_file, file=True)
    best_logger.info(f"Evaluating model: {best_model_file}")
    evaluate(args, best_model_file, best_logger)
    
    # eval 110-122 epoch.pt.tar
    for epoch in range(110, 122):
        model_file = f"{epoch}.pt.tar"
        log_file = os.path.join(args.checkpoint, f"eval_{epoch}.log")
        logger = get_logger(log_file, file=True)
        logger.info(f"Evaluating model: {model_file}")
        evaluate(args, model_file, logger)
