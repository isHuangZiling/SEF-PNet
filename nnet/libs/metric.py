# jyhan@2020

"""
Provided measure metircs:
    speech separation: (w/ & w/o PIT)
        - SDR
        - SDRi
        - SI-SNR
        - SI-SNRi
    speech enhancement:
        - PESQ
        - STOI
"""

import numpy as np

from pesq import pesq
from pystoi.stoi import stoi

from itertools import permutations
from mir_eval.separation import bss_eval_sources

def cal_sisnr(est, ref, remove_dc=True, eps=1e-8):
    """
    Compute SI-SNR
    Arguments:
        est: vector, enhanced/separated signal
        ref: vector, reference signal(ground truth)
    """
    assert len(est) == len(ref)
    def vec_l2norm(x):
        return np.linalg.norm(x, 2)

    # zero mean, seems do not hurt results
    if remove_dc:
        e_zm = est - np.mean(est)
        r_zm = ref - np.mean(ref)
        t = np.inner(e_zm, r_zm) * r_zm / (vec_l2norm(r_zm)**2 + eps)
        n = e_zm - t
    else:
        t = np.inner(est, ref) * ref / (vec_l2norm(ref)**2 + eps)
        n = est - t
    return 20 * np.log10(vec_l2norm(t) / (vec_l2norm(n) + eps))
   

def permute_si_snr(est, ref):
    """
    Compute SI-SNR between N pairs
    Arguments:
        est: list[vector], enhanced/separated signal
        ref: list[vector], reference signal(ground truth)
    Return:
        max sisnr and it's permutation
    """
    assert len(est) == len(ref)
    def si_snr_avg(est, ref):
        return sum([cal_sisnr(e, r) for e, r in zip(est, ref)]) / len(est)

    N = len(est)
    if N != len(est):
        raise RuntimeError(
            "size do not match between est and ref: {:d} vs {:d}".format(
                N, len(ref)))
    si_snrs = []
    perm = []
    for order in permutations(range(N)):
        si_snrs.append(si_snr_avg(est, [ref[n] for n in order]))
        perm.append(order)
    
    return max(si_snrs), perm[si_snrs.index(max(si_snrs))]    


def permute_si_snri(mix, est, ref, both=True):
    """
    Compute SI-SNR improvement
    Arguments:
        mix: vector, mixture signal
        est: list[vector], enhanced/separated signal
        ref: list[vector], reference signal(ground truth)
            [spk1, spk2, aux]
    """    
    m_mix = sum([cal_sisnr(mix, r) for r in ref[:2]]) / len(ref[:2])
    m_enh, _ = permute_si_snr(est, ref)
    if both:
        return m_enh, m_enh - m_mix
    else:
        return m_enh - m_mix

def pit_rank_sisnr(mix, est, ref):
    """
    Compute SI-SNR improvement
    Arguments:
        mix: vector, mixture signal
        est: list[vector], enhanced/separated signal
        ref: list[vector], reference signal(ground truth)
            [spk1, spk2, aux]
    """    
    m_mix1 = sum([cal_sisnr(mix, r) for r in ref[:2]]) / len(ref[:2])
    m_mix2 = sum([cal_sisnr(mix, r) for r in est[:2]]) / len(est[:2])
    m_mix = (m_mix1 + m_mix2) / 2
    m_enh, _ = permute_si_snr(est, ref)
    
    return m_enh, m_mix
    
def pit_rank_sisnr_all(mix, est, ref):
    """
    Compute SI-SNR improvement
    Arguments:
        mix: vector, mixture signal
        est: list[vector], enhanced/separated signal
        ref: list[vector], reference signal(ground truth)
            [spk1, spk2, aux]
    """    
    m_mix1 = sum([cal_sisnr(mix, r) for r in ref[:2]]) / len(ref[:2])
    m_mix2 = sum([cal_sisnr(mix, r) for r in est[:2]]) / len(est[:2])
    m_mix = (m_mix1 + m_mix2) / 2
    m_enh, _ = permute_si_snr(est, ref)
    
    return m_enh, m_mix1, m_mix2, m_mix    


def reorder_list(slist, perm):
    """
    Arguments:
        slist: list[vector], reference signal
        perm: permutation label
    Return:
        list[vector], reordered reference signal
    """
    return [slist[p] for p in perm]


def cal_SDRi(mix, est, ref):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        mix: numpy.ndarray,
        est: [numpy.ndarray, numpy.ndarray] enhanced/separated signal
        ref: [numpy.ndarray, numpy.ndarray] , reference signal(ground truth)
    Returns:
        avg_sdr, sdri
    """
    mix = np.array(mix)
    est = np.array(est)
    ref = np.array(ref)
    
    mix_anchor = np.stack([mix, mix], axis=0)
    sdr, sir, sar, popt = bss_eval_sources(ref, est)
    sdr0, sir0, sar0, popt0 = bss_eval_sources(ref, mix_anchor)
    avg_sdr = (sdr[0] + sdr[1] ) / 2
    avg_sdr_m = (sdr0[0] + sdr0[1] ) / 2 
    
    return avg_sdr, avg_sdr - avg_sdr_m


def permute_pesq(est, ref, fs=8000, mode='nb'):
    """
    Evaluate PESQ
    Args:
        est: [numpy 1D array, numpy 1D array], estimated audio signal 
        ref: [numpy 1D array, numpy 1D array], reference audio signal
        fs:  integer, sampling rate
    """
    assert fs in [8000, 16000]
    assert len(est) == len(ref)
    mode = 'nb' if fs == 8000 else 'wb'

    def pesq_avg(est, ref):
        return sum([pesq(fs, r, e, mode) for e, r in zip(est, ref)]) / len(est)

    N = len(est)
    if N != len(est):
        raise RuntimeError(
            "size do not match between est and ref: {:d} vs {:d}".format(
                N, len(ref)))
    pesqs = []
    for order in permutations(range(N)):
        pesqs.append(pesq_avg(est, [ref[n] for n in order]))
    
    return max(pesqs)  


def permute_stoi(est, ref, fs=8000):
    """
    Evaluate STOI
    Args:
        est: [numpy 1D array, numpy 1D array], estimated audio signal 
        ref: [numpy 1D array, numpy 1D array], reference audio signal
        fs:  integer, sampling rate
    """
    assert len(est) == len(ref)

    def stoi_avg(est, ref):
        return sum([stoi(r, e, fs) for e, r in zip(est, ref)]) / len(est)

    N = len(est)
    if N != len(est):
        raise RuntimeError(
            "size do not match between est and ref: {:d} vs {:d}".format(
                N, len(ref)))
    stois = []
    for order in permutations(range(N)):
        stois.append(stoi_avg(est, [ref[n] for n in order]))
    
    return max(stois)  
    

def eval_all(mix, est, ref, fs=8000, pesq=False):
    """
    Arguments:
        mix: np.narray
        est: list[np.narray, np.narray]
        ref: list[np.narray, np.narray]
    Evaluate 
        SISNR/SISNRi;
        SDR/SDRi;
        PESQ/STOI
    """
    sisnr, sisnri = permute_si_snri(mix, est, ref, True)
    sdr, sdri = cal_SDRi(mix, est, ref)
    if pesq:
        enh_pesq = permute_pesq(est, ref, fs)
        enh_stoi = permute_stoi(est, ref, fs)
        return sisnr, sisnri, sdr, sdri, enh_pesq, enh_stoi
    else:
        return sisnr, sisnri, sdr, sdri

if __name__ == '__main__':
#    np.random.seed(20)
    x = np.random.rand(32000)
    xlist = [np.random.rand(32000), np.random.rand(32000)]
    slist = [np.random.rand(32000), np.random.rand(32000)]
    mlist = [np.random.rand(32000), np.random.rand(32000)]
#    print(permute_si_snr(xlist, slist))
#    print(permute_si_snri(x, xlist, slist))
#    print(permute_si_snri(x, xlist, slist, False))
#    rlist = reorder_list(slist, [0,1])
#    sdr, sir, sar, popt = bss_eval_sou1rces(np.array(slist), np.array(xlist))
#    sdr, sdri = cal_SDRi(x, xlist, slist)
#    pp = permute_pesq(xlist, slist, fs=8000)
#    st = permute_stoi(xlist, xlist, fs=8000)
    sisnr, sisnri, sdr, sdri, enh_pesq, enh_stoi = eval_all(x, xlist, slist, 8000)

#    print(sdr)
#    print(cal_sdr(np.array(xlist[0]), np.array(slist[0])))
#    print(cal_sdr(np.array(xlist[1]), np.array(slist[1])))
#    print(cal_sdr(np.array(xlist[0]), np.array(slist[1])))
#    print(cal_sdr(np.array(xlist[1]), np.array(slist[0])))
#    print(cal_sdr(np.array(xlist), np.array(slist)))
    





