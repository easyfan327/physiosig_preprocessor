from scipy.signal import cheby2
from scipy.signal import sosfiltfilt
from scipy.signal import find_peaks
import numpy as np


def ppg_extract_features_persegment(ppg_seg: np.ndarray, fs: float, fl: float, fh:float):

    sos = cheby2(4, 20, [fl*2/fs, fh*2/fs], 'bandpass', output='sos')
    filtered_ppg_seg = sosfiltfilt(sos, ppg_seg) 
    ppg_seg_mean = np.mean(filtered_ppg_seg)
    ppg_seg_std = np.std(filtered_ppg_seg)

    norm_ppg_seg = (filtered_ppg_seg - ppg_seg_mean) / ppg_seg_std

    ppg_maximas = find_peaks(norm_ppg_seg, prominence=0.1, width=200)
    ppg_minimas = find_peaks(norm_ppg_seg, prominence=0.1, width=100)

if __name__ == "__main__":
    print("Test routine")


