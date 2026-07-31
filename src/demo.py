"""
Real-time style demo: scrolls a raw ECG recording and overlays the model's
predicted beat label at each detected R-peak, using only the from-scratch
Pan-Tompkins detector (no expert annotations), to show the full pipeline
running autonomously the way it would on a wearable device.
"""
import os
import numpy as np
import wfdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow import keras

from preprocessing import bandpass_filter, pan_tompkins_detect, FS, HALF

RECORD = "208"  # a record with a good mix of Normal and Ventricular ectopic beats
DATA_DIR = "data/mitdb"
RESULTS_DIR = "results"
FIG_DIR = "figures"
CLASSES = ["N", "S", "V", "F", "Q"]
WINDOW_SECONDS = 4  # width of the scrolling view


def build_beats(filtered, peaks):
    beats, valid_peaks = [], []
    for peak in peaks:
        if peak - HALF < 0 or peak + HALF > len(filtered):
            continue
        segment = filtered[peak - HALF: peak + HALF]
        seg_range = segment.max() - segment.min()
        if seg_range == 0:
            continue
        segment = (segment - segment.min()) / seg_range
        beats.append(segment.astype(np.float32))
        valid_peaks.append(peak)
    return np.array(beats), np.array(valid_peaks)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    record = wfdb.rdrecord(os.path.join(DATA_DIR, RECORD))
    raw_signal = record.p_signal[:, 0]
    filtered = bandpass_filter(raw_signal)

    peaks = pan_tompkins_detect(raw_signal)
    beats, valid_peaks = build_beats(filtered, peaks)
    print(f"Detected {len(valid_peaks)} beats in record {RECORD} using our own R-peak detector.")

    model = keras.models.load_model(os.path.join(RESULTS_DIR, "ecg_1dcnn_model.keras"))
    probs = model.predict(beats[..., np.newaxis], verbose=0)
    predictions = np.argmax(probs, axis=1)

    window_samples = WINDOW_SECONDS * FS
    total_frames = 200
    step = max(1, (len(filtered) - window_samples) // total_frames)

    fig, ax = plt.subplots(figsize=(10, 4))
    line, = ax.plot([], [], color="tab:blue", linewidth=1)
    scatter = ax.scatter([], [], color="tab:red", zorder=5)
    label_texts = []
    ax.set_xlim(0, WINDOW_SECONDS)
    ax.set_ylim(filtered.min() * 1.1, filtered.max() * 1.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (normalized)")
    ax.set_title(f"Live ECG Classification Demo, Record {RECORD}")

    def init():
        line.set_data([], [])
        scatter.set_offsets(np.empty((0, 2)))
        return line, scatter

    def update(frame):
        start = frame * step
        end = start + window_samples
        if end > len(filtered):
            return line, scatter
        t = np.arange(start, end) / FS - start / FS
        line.set_data(t, filtered[start:end])

        for txt in label_texts:
            txt.remove()
        label_texts.clear()

        in_window = (valid_peaks >= start) & (valid_peaks < end)
        peak_times = (valid_peaks[in_window] - start) / FS
        peak_vals = filtered[valid_peaks[in_window]]
        scatter.set_offsets(np.column_stack([peak_times, peak_vals]) if len(peak_times) else np.empty((0, 2)))

        for pt, pv, pred_idx in zip(peak_times, peak_vals, predictions[in_window]):
            txt = ax.annotate(CLASSES[pred_idx], (pt, pv), textcoords="offset points",
                               xytext=(0, 12), ha="center", fontsize=11, fontweight="bold",
                               color="tab:red")
            label_texts.append(txt)
        return line, scatter

    ani = animation.FuncAnimation(fig, update, frames=total_frames, init_func=init,
                                   interval=80, blit=False)
    out_path = os.path.join(FIG_DIR, "ecg_live_demo.gif")
    ani.save(out_path, writer="pillow", fps=12)
    print(f"Demo animation saved to {out_path}")


if __name__ == "__main__":
    main()
