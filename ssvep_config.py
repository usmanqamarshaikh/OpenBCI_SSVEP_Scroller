# ─── USER‑TUNABLE VALUES ────────────────────────────────
# Stimulus
FREQ_UP     = 10          # Hz  (top window)
FREQ_DOWN   = 13          # Hz  (bottom window)
STIM_SIZE   = 250         # px
WINDOW_SIZE = (300, 300)  # per SDL2 window

# Calibration
BLOCK_SEC   = 10
REST_SEC    = 4
NUM_CYCLES  = 3


# Acquisition
FS          = 250         # Hz
CH_O1, CH_O2 = 0, 2       # indices in the OpenBCI stream

# Feature extraction
EPOCH_SEC   = 3           # window length
HOP_SEC     = 0.25
BAND_HZ     = 0.5         # ± band for power
NOISE_HZ    = 2.0         # ± band for noise

# Output keys
KEY_UP      = 'pageup'
KEY_DOWN    = 'pagedown'

# File paths
CALIB_FILE  = "calib_data.npz"
MODEL_FILE  = "svm.pkl"   # saved inside repo root
# ─────────────────────────────────────────────────────────
