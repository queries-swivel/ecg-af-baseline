import csv
import math
import os
from datetime import datetime
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import wfdb
from matplotlib.backends.backend_pdf import PdfPages

# --- Configurable layout ---
LEAD_ORDER = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
ROWS, COLS = 2, 6       # 12 leads grid
SECONDS = 10.0          # length per page
MM_PER_MV = 10.0        # calibration
MM_PER_S  = 25.0        # paper speed
DPI = 300               # print quality
MARG_MM = 8.0           # page margin

def mm_to_inches(mm: float) -> float:
    return mm / 25.4

def ecg_grid(ax, x_seconds: float, y_mv: float):
    """Draw classic ECG grid: small 1mm, big 5mm boxes."""
    # figure coordinates are in inches; create a grid in mm via axis limits
    ax.set_facecolor("white")
    ax.set_axis_off()
    xmin, xmax = 0, x_seconds
    ymin, ymax = -y_mv, y_mv
    ax.set_xlim(0, x_seconds)
    ax.set_ylim(-y_mv, y_mv)

    # Small grid: 1 mm → convert to seconds/mV spacing
    small_t = 1.0 / MM_PER_S            # seconds per 1 mm
    small_v = 1.0 / MM_PER_MV           # mV per 1 mm

    # Thin lines
    for t in np.arange(0, x_seconds + 1e-6, small_t):
        ax.axvline(t, linewidth=0.3, alpha=0.3)
    for v in np.arange(-y_mv, y_mv + 1e-6, small_v):
        ax.axhline(v, linewidth=0.3, alpha=0.3)

    # Bold every 5mm
    for t in np.arange(0, x_seconds + 1e-6, small_t * 5):
        ax.axvline(t, linewidth=0.8, alpha=0.6)
    for v in np.arange(-y_mv, y_mv + 1e-6, small_v * 5):
        ax.axhline(v, linewidth=0.8, alpha=0.6)

def calibrate_trace(sig: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (t, mv) for exactly SECONDS duration (pad/trim)."""
    n = int(round(SECONDS * fs))
    s = sig.copy()
    if s.ndim > 1:
        # pick column 0 if 2D slipped through
        s = s[:, 0]
    if len(s) < n:
        s = np.pad(s, (0, n - len(s)), constant_values=np.nan)
    else:
        s = s[:n]
    t = np.arange(len(s)) / fs
    return t, s

def normalise_units(record) -> float:
    """
    Try to infer units from WFDB header.
    Returns scale factor to convert to mV if needed.
    """
    # WFDB stores units in record.sig_info[i]['units'] when available
    # Many public sets already in mV; be conservative
    return 1.0

def plot_lead(ax, t, mv, label: str):
    # Avoid connecting NaN gaps
    ax.plot(t, mv, linewidth=0.9)
    ax.text(0.2, 0.85, label, transform=ax.transAxes, fontsize=8, weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

def load_12lead(record_path: str) -> Tuple[np.ndarray, int, List[str]]:
    """
    Load WFDB record and return (signals[LxN], fs, lead_names).
    """
    rec = wfdb.rdsamp(record_path)
    sig = rec.p_signals  # in physical units
    fs = int(rec.fs)
    # Build lead map
    if hasattr(rec, "sig_name") and rec.sig_name:
        names = list(rec.sig_name)
    else:
        names = [f"L{i+1}" for i in range(sig.shape[1])]
    return sig.T, fs, names  # L x N

def reorder_leads(sig_LxN: np.ndarray, names: List[str]) -> Tuple[np.ndarray, List[str]]:
    name_to_idx = {n.upper(): i for i, n in enumerate(names)}
    idx = []
    actual = []
    for want in LEAD_ORDER:
        if want.upper() in name_to_idx:
            idx.append(name_to_idx[want.upper()])
            actual.append(want)
    arr = sig_LxN[idx, :] if idx else sig_LxN
    names_out = actual if idx else names
    return arr, names_out

def make_page(fig, axes, record_meta, sig_LxN, fs):
    # header
    fig.suptitle(
        f"{record_meta['dataset']} | {os.path.basename(record_meta['record_path'])} | "
        f"Label: {record_meta['label']} | fs={fs} Hz | Age={record_meta.get('age','?')} | Sex={record_meta.get('sex','?')}",
        fontsize=10
    )
    # per subplot grid + trace
    y_mv = 1.5  # display range per lead (+/- mV)
    for i, ax in enumerate(axes.flat):
        if i >= sig_LxN.shape[0]:
            ax.axis("off")
            continue
        t, s = calibrate_trace(sig_LxN[i], fs)
        # assume already mV (if your signals are in volts, scale here)
        ecg_grid(ax, SECONDS, y_mv)
        plot_lead(ax, t, s, LEAD_ORDER[i] if i < len(LEAD_ORDER) else f"L{i+1}")

def generate_booklet(csv_path: str, out_pdf: str):
    A4_W_MM, A4_H_MM = 210, 297
    fig_w = mm_to_inches(A4_W_MM - 2*MARG_MM)
    fig_h = mm_to_inches(A4_H_MM - 2*MARG_MM)

    with PdfPages(out_pdf) as pdf:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                record_path = row["record_path"]
                try:
                    sig_LxN, fs, names = load_12lead(record_path)
                    sig_LxN, names = reorder_leads(sig_LxN, names)

                    fig, axes = plt.subplots(ROWS, COLS, figsize=(fig_w, fig_h), dpi=DPI)
                    make_page(fig, axes, row, sig_LxN, fs)

                    # footer
                    dt = datetime.now().strftime("%Y-%m-%d %H:%M")
                    fig.text(0.01, 0.01, f"Generated {dt}  ·  Paper speed {MM_PER_S} mm/s  ·  Gain {MM_PER_MV} mm/mV",
                             fontsize=7, alpha=0.7)
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    pdf.savefig(fig, dpi=DPI)
                    plt.close(fig)
                except Exception as e:
                    print(f"[WARN] Skipping {record_path}: {e}")

if __name__ == "__main__":
    in_csv = "data/interim/af_examples.csv"
    out_pdf = "reports/AF_examples_v1.pdf"
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    generate_booklet(in_csv, out_pdf)
    print(f"Written {out_pdf}")

