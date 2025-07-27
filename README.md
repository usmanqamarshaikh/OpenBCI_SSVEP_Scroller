# OPENBCI-SSVEP‑Scroller 🧠

A tiny Python brain‑computer interface that lets you **scroll up or down with nothing but your gaze**:

1. Focus the **top flickering square** → sends the ↑ key  
2. Focus the **bottom flickering square** → sends the ↓ key  
3. Look away → idle

Two occipital electrodes (O1 & O2) + an OpenBCI Cyton are all you need.

---

## 1 · Folder layout

├─ ssvep_config.py # <‑‑‑ edit to change settings.

├─ block1_calibrate.py # record calibration data (flicker only).

├─ block2_train.py # extract features, update svm.pkl.

├─ block3_live.py # real‑time control.

├─ requirements.txt # pip dependencies.

└─ README.md

## 2 · Quick start
```bash
git clone https://github.com/usmanqamarshaikh/OBCI_SSVEP_Scroller.git
cd OBCI_SSVEP_Scroller
pip install -r requirements.txt
```
## 3 · 3 Steps

### 1. OpenBCI setup.
1. Put electrodes at O1 & O2.
2. start OpenBCI‑GUI.
3. enable LSL stream of "TimeSeriesFilt" from networking tab.


### 2. Calibrate
run 'block1_calibrate.py' or run the following command.
```bash
python block1_calibrate.py
```
Focus the squares during the calibratioon session. try not to move.
once the data calibration session ends, a file calib_data.npz will be saved in the root directory.

run 'block2_train.py' or run the following command.
```bash
python block2_train.py
```

### 3. Run!
run 'block1_live.py' or run the following command.
```bash
python block3_live.py
```
Move the two stimulus windows to comfortable positions. Focus on the top square to scroll up, bottom square to scroll down.


## Customize the settings
Open ssvep_config.py and change anything. Following are the initial values.

FREQ_UP   = 10          # Hz
FREQ_DOWN = 13
EPOCH_SEC = 3           # window length
KEY_UP    = 'pageup'
KEY_DOWN  = 'pagedown'
WINDOW_SIZE = (300, 300)

## How it works

| Stage               | Description                                                                                                                               |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Stimulus**        | Two independent SDL2 windows flicker at `FREQ_UP` / `FREQ_DOWN` frequency.                                                                |
| **Acquisition**     | OpenBCI‑GUI band‑passes (5–20 Hz) and streams via LSL; `block3_live.py` keeps a 3‑s ring buffer (adjustable).                             |
| **Features (18 D)** | *Per epoch*: CCA correlation at f₁ & f₂, plus Welch power & SNR at f₁, f₂ and 2× harmonics for each channel.                              |
| **Classifier**      | Linear SVM incrementally refit after each calibration session (`block2_train.py`).                                                        |
| **Control**         | Prediction every 250 ms (adjustable); on label change sends `cfg.KEY_UP` / `cfg.KEY_DOWN` via PyAutoGUI.                                  |
