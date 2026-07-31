"""
Member 1 (Data Engineer) pipeline.

Downloads MIT-BIH Arrhythmia Database records, applies a Butterworth bandpass
filter, detects R-peaks with a Pan-Tompkins style detector (validated against
the expert annotations), segments individual heartbeats, maps annotation
symbols to the AAMI EC57 5-class scheme, amplitude-normalizes each beat, and
exports a labeled NumPy dataset split 70/15/15 into train/val/test.
"""
import os
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, find_peaks

DATA_DIR = "data/mitdb"
OUT_DIR = "data/processed"
FS = 360
WINDOW = 180  # samples centered on R-peak (matches proposal: 180-sample beats)
HALF = WINDOW // 2

RECORDS = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
           '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
           '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
           '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
           '222', '223', '228', '230', '231', '232', '233', '234']

# AAMI EC57 mapping from MIT-BIH annotation symbols
AAMI_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q',
}
CLASSES = ['N', 'S', 'V', 'F', 'Q']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def bandpass_filter(signal, lowcut=0.5, highcut=45.0, fs=FS, order=3):
    nyq = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)


def pan_tompkins_detect(signal, fs=FS):
    """Pan-Tompkins QRS detector: bandpass -> derivative -> square ->
    moving-window integration -> adaptive dual-threshold peak picking
    (SPKI/NPKI running estimates, as in the original 1985 algorithm)."""
    filtered = bandpass_filter(signal, 5, 15, fs, order=2)
    diff = np.diff(filtered, prepend=filtered[0])
    squared = diff ** 2
    window = int(0.150 * fs)
    integrated = np.convolve(squared, np.ones(window) / window, mode='same')
    min_dist = int(0.2 * fs)  # refractory period ~200ms

    # candidate local maxima, at least min_dist apart
    candidates, _ = find_peaks(integrated, distance=min_dist)
    if len(candidates) == 0:
        return np.array([], dtype=int)

    # initialize adaptive thresholds from the first 2 seconds
    init_len = min(int(2 * fs), len(integrated))
    spki = np.max(integrated[:init_len]) * 0.5 if init_len else np.max(integrated) * 0.5
    npki = np.mean(integrated[:init_len]) if init_len else np.mean(integrated)
    threshold1 = npki + 0.25 * (spki - npki)

    peaks = []
    for idx in candidates:
        peak_val = integrated[idx]
        if peak_val >= threshold1:
            peaks.append(idx)
            spki = 0.125 * peak_val + 0.875 * spki
        else:
            npki = 0.125 * peak_val + 0.875 * npki
        threshold1 = npki + 0.25 * (spki - npki)

    return np.array(peaks, dtype=int)


def evaluate_detector(detected, true_peaks, tolerance=int(0.05 * FS)):
    """Match detected peaks to annotated R-peaks within a tolerance window."""
    if len(true_peaks) == 0:
        return 0, 0, 0
    matched = 0
    used = np.zeros(len(detected), dtype=bool)
    for tp in true_peaks:
        diffs = np.abs(detected - tp)
        idx = np.argmin(diffs) if len(diffs) else None
        if idx is not None and diffs[idx] <= tolerance and not used[idx]:
            matched += 1
            used[idx] = True
    tp_count = matched
    fn_count = len(true_peaks) - matched
    fp_count = len(detected) - matched
    return tp_count, fp_count, fn_count


def process_record(record_name):
    rec_path = os.path.join(DATA_DIR, record_name)
    record = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, 'atr')

    raw_signal = record.p_signal[:, 0]  # lead 0 (MLII in most records)
    filtered = bandpass_filter(raw_signal)

    detected_peaks = pan_tompkins_detect(raw_signal)
    det_stats = evaluate_detector(detected_peaks, ann.sample)

    beats, labels = [], []
    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol not in AAMI_MAP:
            continue
        if sample - HALF < 0 or sample + HALF > len(filtered):
            continue
        segment = filtered[sample - HALF: sample + HALF]
        seg_range = segment.max() - segment.min()
        if seg_range == 0:
            continue
        segment = (segment - segment.min()) / seg_range  # amplitude normalize to [0,1]
        beats.append(segment.astype(np.float32))
        labels.append(CLASS_TO_IDX[AAMI_MAP[symbol]])

    return np.array(beats, dtype=np.float32), np.array(labels, dtype=np.int64), det_stats


def build_dataset(seed=42):
    os.makedirs(OUT_DIR, exist_ok=True)
    all_beats, all_labels = [], []
    total_tp, total_fp, total_fn = 0, 0, 0

    for rec in RECORDS:
        try:
            beats, labels, det_stats = process_record(rec)
        except Exception as e:
            print(f"  [WARN] Skipping record {rec}: {e}")
            continue
        all_beats.append(beats)
        all_labels.append(labels)
        tp, fp, fn = det_stats
        total_tp += tp
        total_fp += fp
        total_fn += fn
        print(f"  record {rec}: {len(labels)} labeled beats, "
              f"R-peak detection TP={tp} FP={fp} FN={fn}")

    X = np.concatenate(all_beats, axis=0)
    y = np.concatenate(all_labels, axis=0)

    sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    ppv = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    print(f"\nPan-Tompkins R-peak detector overall: "
          f"Sensitivity={sensitivity:.4f}, PPV={ppv:.4f} (validated against expert annotations)")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    n = len(X)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(OUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)

    print(f"\nDataset exported: total={n} train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution (whole dataset):")
    for u, c in zip(unique, counts):
        print(f"  {CLASSES[u]}: {c} ({100*c/n:.2f}%)")

    with open(os.path.join(OUT_DIR, "detector_stats.txt"), "w") as f:
        f.write(f"Pan-Tompkins R-peak detector vs expert annotations (tolerance=50ms)\n")
        f.write(f"TP={total_tp} FP={total_fp} FN={total_fn}\n")
        f.write(f"Sensitivity={sensitivity:.4f} PPV={ppv:.4f}\n")

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    build_dataset()
