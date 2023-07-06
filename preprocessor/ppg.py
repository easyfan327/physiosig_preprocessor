import traceback
from scipy.signal import cheby2
from scipy.signal import sosfiltfilt
from scipy.signal import find_peaks
from scipy.fft import fft, ifft
from scipy.signal import decimate
from scipy.signal.windows import blackman
import string
import random
import numpy as np
import matplotlib.pyplot as plt
import pprint

from libs import smooth

def ppg_hr_estimate(ppg_seg: np.ndarray, fs:float, verbose=False):
    N = len(ppg_seg)
    window = blackman(N)

    fft_N = int(2**(np.ceil(np.log2(N)) + 2))

    spec = np.abs(fft(ppg_seg * window, fft_N))[0: fft_N//2]
    freq = np.arange(start=0,stop=fs/2,step=fs/fft_N)
    spec_maximas = find_peaks(spec, height=np.max(spec)/4)
 
    f_candicate_N = len(spec_maximas[0])
    if f_candicate_N == 0:
        best_idx = None
    else:
        cred = dict()

        best_idx = 0
        best_cred = 0

        for idx, f, s in zip(spec_maximas[0], freq[spec_maximas[0]], spec[spec_maximas[0]]):
            h_count = 1
            if np.abs(np.min(f * 2 - freq[spec_maximas[0]])) < 0.05:
                h_count = h_count + 1
            if np.abs(np.min(f * 3 - freq[spec_maximas[0]])) < 0.05:
                h_count = h_count + 1
            cred[idx]= s * h_count
            if cred[idx] > best_cred:
                best_cred = cred[idx]
                best_idx = idx
    
    if verbose:
        print("using {:d} point FFT".format(fft_N))
        fig, axe = plt.subplots(2, 1)
        axe[0].plot(ppg_seg)
        axe[1].plot(freq, spec)
        axe[1].set_xlabel("freq")
        axe[1].set_ylabel("amp")
        axe[1].scatter(freq[spec_maximas[0]], spec[spec_maximas[0]])
        print(cred)
        for i, c in cred.items():
            print(c, i, freq[i], spec[i])
            axe[1].annotate("{:4.2f}".format(c), (freq[i], spec[i]))
        fig.savefig("debug-fft.png")

    if best_idx is not None:
        return freq[best_idx]
    else:
        return None

def ppg_extract_fiducial_points_percycle(ppg_cycle: np.ndarray, fs:int, s_rel:int, seg_id:int, cycle_id:int, verbose=False):
    # s_rel: relative index of systolic peak
    feature = dict()
    T = len(ppg_cycle)
    ppg = ppg_cycle

    #vpg = smooth(np.diff(ppg), 3, 'flat')
    #apg = smooth(np.diff(vpg), 5, 'flat')
    #jpg = smooth(np.diff(apg), 7, 'flat')

    vpg = np.convolve(np.diff(ppg), np.ones(3), 'same') / 3
    apg = np.convolve(np.diff(vpg), np.ones(5), 'same') / 5
    jpg = np.convolve(np.diff(apg), np.ones(7), 'same') / 7

    print(len(ppg))
    print(len(vpg))
    print(len(apg))
    print(len(jpg))
 
    vpg_maximas = find_peaks(vpg, prominence=0.1)
    max_acc_idx = vpg_maximas[0][np.argmax(vpg_maximas[1]['prominences'])]
    print("max_acc_idx", max_acc_idx)
    apg_maximas = find_peaks(apg)
    jpg_maximas = find_peaks(jpg)
    vpg_minimas = find_peaks(-vpg, prominence=0.1)
    apg_minimas = find_peaks(-apg)
    jpg_minimas = find_peaks(-jpg)
    apg_zerocrossings = np.where(np.diff(np.sign(apg)))[0]
    jpg_zerocrossings = np.where(np.diff(np.sign(jpg)))[0]

    def remove_peaks(extrema, left_bound):
        index_to_be_removed = list()
        for i in range(len(extrema[0])):
            idx = extrema[0][i]
            if idx <= left_bound:
                index_to_be_removed.append(i)

        index_to_be_removed = np.asarray(index_to_be_removed)
        print("index_to_be_removed", index_to_be_removed)
        if len(index_to_be_removed) > 0:
            arr = np.delete(extrema[0], index_to_be_removed)
            data = dict()
            for key, value in extrema[1].items():
                data[key] = np.delete(extrema[1][key], index_to_be_removed)
        
            #extrema is tuple -- non-mutable
            return (arr, data)
        else:
            return extrema
    
    left_bound_idx = 0
    for i in range(len(vpg_minimas[0])):
        idx = vpg_minimas[0][i]
        if idx < max_acc_idx:
            left_bound_idx = idx
    
    if left_bound_idx > 0:
        vpg_maximas = remove_peaks(vpg_maximas, left_bound_idx)
        vpg_minimas = remove_peaks(vpg_minimas, left_bound_idx)
        apg_maximas = remove_peaks(apg_maximas, left_bound_idx)
        apg_minimas = remove_peaks(apg_minimas, left_bound_idx)
        jpg_maximas = remove_peaks(jpg_maximas, left_bound_idx)
        jpg_minimas = remove_peaks(jpg_minimas, left_bound_idx)
        apg_zerocrossings = np.delete(apg_zerocrossings, np.where((apg_zerocrossings - left_bound_idx)<=0))
        jpg_zerocrossings = np.delete(jpg_zerocrossings, np.where((jpg_zerocrossings - left_bound_idx)<=0))
    
    a = apg_maximas[0][0]
    b = apg_minimas[0][0]

    if apg[apg_maximas[0][1]] > 0 or apg_maximas[0][1] - apg_minimas[0][1] > np.floor(0.1 * fs):
        # c, d not present in APG or c, d not prominent

        if jpg[jpg_minimas[0][1]] < 0:
            # c, d not present
            if jpg_maximas[0][0] > jpg_minimas[0][0]:
                c = jpg_maximas[0][0]
            elif jpg_maximas[0][1] > jpg_minimas[0][0]:
                c = jpg_maximas[0][1]
            elif jpg_maximas[0][2] > jpg_minimas[0][0]:
                c = jpg_maximas[0][2]
            d = apg_zerocrossings[1]
            case_name = "CASE I"
        else:
            # c, d not prominent
            if apg_maximas[0][1] < 0:
                c = apg_maximas[0][1]
                d = apg_minimas[0][1]
                case_name = "CASE II-1"
            else:
                c = int(jpg_minimas[0][1] - np.floor(T / 100 * 2.5))
                d = int(jpg_minimas[0][1] + np.floor(T / 100 * 2.5))
                case_name = "CASE II-2"

        e = jpg_zerocrossings[2]
        try:
            f = jpg_zerocrossings[3]
        except:
            traceback.print_exc()
            f = 0
    else:
        # c, d present in APG
        c = apg_maximas[0][1]
        d = apg_minimas[0][1]
        e = apg_maximas[0][2]
        f = apg_minimas[0][2]
        case_name = "CASE III"

    notch = e
    diastolic = f

    f_pts = {
        'a': a,
        'b': b,
        'c': c,
        'd': d,
        'e': e,
        'f': f
    }
    pprint.pprint(f_pts)
    if verbose:
        fig, axe = plt.subplots(1, 4, figsize=(24, 6), dpi=144)
        try:
            fig.suptitle(case_name)
            axe[0].plot(ppg, c='royalblue')
            axe[0].scatter(s_rel, ppg[s_rel])
            axe[0].axhline(0, linestyle=':', c='black')
            axe[1].plot(vpg, c='royalblue')
            axe[1].scatter(vpg_maximas[0], vpg[vpg_maximas[0]], s=50, linewidth=4, marker='+', c='navy')
            axe[1].scatter(vpg_minimas[0], vpg[vpg_minimas[0]], s=50, linewidth=4, marker='x', c='orange')
            axe[1].set_title("VPG")
            pprint.pprint(vpg_maximas)
            axe[1].axhline(0, linestyle=':', c='black')
            axe[2].plot(apg, c='royalblue')
            axe[2].scatter(apg_maximas[0], apg[apg_maximas[0]], s=50, linewidth=4, marker='+', c='navy')
            axe[2].scatter(apg_minimas[0], apg[apg_minimas[0]], s=50, linewidth=4, marker='x', c='orange')
            axe[2].scatter(apg_zerocrossings, apg[apg_zerocrossings], s=50, linewidth=4, marker='x', c='forestgreen')
            axe[2].set_title("APG")
            axe[2].axhline(0, linestyle=':', c='black')
            for key, value in f_pts.items():
                axe[2].scatter(value, apg[value], s=200, linewidth=2, marker='+', c='brown')
                #axe[2].axvline(value, linestyle=':', c='brown')
                axe[2].annotate(key, (value + 1, apg[value]), fontsize='x-large')
            axe[3].plot(jpg, c='royalblue')
            axe[3].scatter(jpg_maximas[0], jpg[jpg_maximas[0]], s=50, linewidth=4, marker='+', c='navy')
            axe[3].scatter(jpg_minimas[0], jpg[jpg_minimas[0]], s=50, linewidth=4, marker='x', c='orange')
            axe[3].scatter(jpg_zerocrossings, jpg[jpg_zerocrossings], s=50, linewidth=4, marker='x', c='forestgreen')
            axe[3].set_title("JPG")
            axe[3].axhline(0, linestyle=':', c='black')
        except:
            traceback.print_exc()
        finally: 
            fig.savefig("features-{:4d}-{:2d}.png".format(seg_id, cycle_id))
    
    return f_pts

def ppg_extract_features_persegment(ppg_seg: np.ndarray, fs: float, fl: float, fh:float, decimate_q: int, seg_id: int, verbose: bool):

    sos = cheby2(4, 20, [fl*2/fs, fh*2/fs], 'bandpass', output='sos')
    filtered_ppg_seg = sosfiltfilt(sos, ppg_seg) 
    ppg_seg_mean = np.mean(filtered_ppg_seg)
    ppg_seg_std = np.std(filtered_ppg_seg)

    norm_ppg_seg = (filtered_ppg_seg - ppg_seg_mean) / ppg_seg_std

    hr_freq = ppg_hr_estimate(decimate(norm_ppg_seg, decimate_q), fs//decimate_q, True)
    dynamic_range = np.abs(np.max(norm_ppg_seg) - np.min(norm_ppg_seg))

    if hr_freq is not None:
        min_distance = int(0.5 * fs / hr_freq)
        hr_cycle_estimate = fs / hr_freq
    
    if verbose:
        print("hr_freq = {:4.2f}".format(hr_freq)) 
        print("min_distance = {:d}".format(min_distance)) 
        print("dynamic_range = {:4.2f}".format(dynamic_range)) 
    
    ppg_maximas = find_peaks(norm_ppg_seg, prominence=dynamic_range/8, distance=min_distance)
    ppg_minimas = find_peaks(-norm_ppg_seg, prominence=dynamic_range/8, distance=min_distance)

    if verbose:
        fig, axe = plt.subplots(1, 1)
        axe.plot(norm_ppg_seg)
        axe.scatter(ppg_maximas[0], norm_ppg_seg[ppg_maximas[0]])
        for txt, x, y in zip(ppg_maximas[1]['prominences'], ppg_maximas[0], norm_ppg_seg[ppg_maximas[0]]):
            axe.annotate("prom = {:4.2f}".format(txt), (x, y))
        axe.scatter(ppg_minimas[0], norm_ppg_seg[ppg_minimas[0]])
        for txt, x, y in zip(ppg_minimas[1]['prominences'], ppg_minimas[0], norm_ppg_seg[ppg_minimas[0]]):
            axe.annotate("prom = {:4.2f}".format(txt), (x, y))
        axe.set_title("drange = {:4.2f}, min_dist = {:4.2f}".format(dynamic_range, min_distance))
        fig.savefig("debug-{:4d}.png".format(seg_id))

    if len(ppg_minimas[0]) < 2:
        # no cycles
        return None
    else:
        features = list()
        for i in range(len(ppg_minimas[0]) - 1):
            onset = ppg_minimas[0][i]
            onset_next = ppg_minimas[0][i+1]
            systolic = None

            # the first maxima between two onset is the systolic peak
            for j in ppg_maximas[0]:
                if j > onset and j < onset_next:
                    systolic = j
                    break
            
            if systolic is not None:
                #extend_t = int(np.floor(0.001 * fs))
                extend_t = 1

                features.append(ppg_extract_fiducial_points_percycle(filtered_ppg_seg[onset - extend_t :onset_next + extend_t], fs=fs, s_rel=systolic-onset, seg_id=seg_id, cycle_id=i, verbose=True))
            else:
                break
        return features


if __name__ == "__main__":
    arr = np.loadtxt("ppg.csv", delimiter=",", dtype=float)
    for i in range(19, 20):
        s = decimate(arr[i, :], 10)
        ppg_extract_features_persegment(s, fs=100, fl=0.4, fh=8, decimate_q=5, seg_id=i, verbose=True)

