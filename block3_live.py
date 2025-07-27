#!/usr/bin/env python3
"""
block3_live.py
──────────────────────────────────────────────────────────────
Real‑time SSVEP control

• Imports all user‑tunable values from ssvep_config.py
• Loads the pooled SVM bundle in cfg.MODEL_FILE
• Two SDL2 windows, each filled by a Renderer
      top  window flickers at cfg.FREQ_UP   (→ “Up” key)
      bottom window flickers at cfg.FREQ_DOWN (→ “Down” key)
• 18‑feature pipeline (same as Block 2)
• Sends cfg.KEY_UP / cfg.KEY_DOWN when the predicted
  class changes; “Idle” sends nothing.

Tested on pygame‑SDL2 ≥ 2.1 under Py 3.10 / Windows 11.
"""
import sys, time, threading
import numpy as np, pygame, pyautogui, joblib
import pygame._sdl2 as sdl2
from scipy.signal.windows import hann
from scipy.signal import welch
from sklearn.cross_decomposition import CCA
from pylsl import StreamInlet, resolve_byprop
import ssvep_config as cfg

# ─── constants from config ─────────────────────────────────────────────
FS               = cfg.FS
WIN_SEC          = cfg.EPOCH_SEC
STEP_SEC         = cfg.HOP_SEC
WIN_LEN          = int(FS * WIN_SEC)
FREQ_UP          = cfg.FREQ_UP
FREQ_DOWN        = cfg.FREQ_DOWN
H1, H2           = 2*FREQ_UP, 2*FREQ_DOWN
CH_O1, CH_O2     = cfg.CH_O1, cfg.CH_O2
KEY_UP, KEY_DOWN = cfg.KEY_UP, cfg.KEY_DOWN
WIN_SIZE         = cfg.WINDOW_SIZE
WHITE, GREY      = (255,255,255,255), (120,120,120,255)
# ───────────────────────────────────────────────────────────────────────

# ═════ load model ═════
bundle = joblib.load(cfg.MODEL_FILE)
svm, scaler = bundle['model'], bundle['scaler']
print("Loaded model:", cfg.MODEL_FILE)

# ═════ connect LSL stream ═════
stre = resolve_byprop('type','EEG', timeout=5)
if not stre: sys.exit("❌  No LSL EEG stream.")
inlet = StreamInlet(stre[0], max_buflen=1)
print("Connected to", stre[0].name())

# ring buffer
ring = np.zeros((WIN_LEN,2)); w_idx=0; lock=threading.Lock()
def reader():
    global w_idx
    while True:
        chunk,_ = inlet.pull_chunk(timeout=0.1)
        if chunk:
            d=np.asarray(chunk)[:,[CH_O1,CH_O2]]
            with lock:
                for r in d:
                    ring[w_idx]=r; w_idx=(w_idx+1)%WIN_LEN
threading.Thread(target=reader,daemon=True).start()

# ═════ feature helpers ═════
hann_win = hann(WIN_LEN, sym=False)
def welch_ps(sig): return welch(sig, FS, 'hann', nperseg=len(sig), noverlap=0)
def bp(ps,fv,c):   return ps[(fv>=c-cfg.BAND_HZ)&(fv<=c+cfg.BAND_HZ)].mean()
def npow(ps,fv,c): return ps[((fv>=c-cfg.NOISE_HZ)&(fv<c-0.5))|
                             ((fv>c+0.5)&(fv<=c+cfg.NOISE_HZ))].mean()
def snr(ps,fv,c):  return 10*np.log10(bp(ps,fv,c)/(npow(ps,fv,c)+1e-12))

EPS = 1e-12
def safe_cca_corr(epoch, freq, fs=cfg.FS):
    """1‑component CCA correlation, returns 0 if epoch is flat/invalid."""
    if not np.isfinite(epoch).all() or np.std(epoch) < EPS:
        return 0.0
    t = np.arange(len(epoch)) / fs
    ref = np.column_stack([np.sin(2*np.pi*freq*t), np.cos(2*np.pi*freq*t),
                           np.sin(4*np.pi*freq*t), np.cos(4*np.pi*freq*t)])
    try:
        cca = CCA(1).fit(epoch[:, None], ref)
        u, v = cca.transform(epoch[:, None], ref)
        return float(np.corrcoef(u[:, 0], v[:, 0])[0, 1])
    except Exception:
        return 0.0

def make_feat(o1,o2):
    avg=(o1+o2)/2
    f=[safe_cca_corr(avg,FREQ_UP), safe_cca_corr(avg,FREQ_DOWN)]
    for sig in (o1,o2):
        fv,ps=welch_ps(sig)
        f += [bp(ps,fv,FREQ_UP),   snr(ps,fv,FREQ_UP),
              bp(ps,fv,FREQ_DOWN), snr(ps,fv,FREQ_DOWN),
              bp(ps,fv,H1),        snr(ps,fv,H1),
              bp(ps,fv,H2),        snr(ps,fv,H2)]
    return scaler.transform([f])

# ═════ SDL2 windows ═════
pygame.init()
up_w = sdl2.Window(f"Up {FREQ_UP:.2f} Hz | Idle",   size=WIN_SIZE, position=(100,100))
dn_w = sdl2.Window(f"Down {FREQ_DOWN:.2f} Hz | Idle",size=WIN_SIZE, position=(450,100))
up_r = sdl2.Renderer(up_w);  dn_r = sdl2.Renderer(dn_w)
def fill_square(rend, col): rend.draw_color=col; rend.clear(); rend.present()

# ═════ main loop ═════
pred='Idle'; last=time.time()
while True:
    # ----- inference every hop -----
    if time.time()-last >= STEP_SEC:
        last=time.time()
        with lock:
            buf = np.r_[ring[w_idx:], ring[:w_idx]]
        if len(buf)==WIN_LEN:
            o1,o2 = buf[:,0], buf[:,1]
            lbl = {0:'Up',1:'Down',2:'Idle'}.get(
                    int(svm.predict(make_feat(o1,o2))[0]), 'Idle')
            if lbl!=pred:
                if lbl=='Up':   pyautogui.press(KEY_UP)
                elif lbl=='Down': pyautogui.press(KEY_DOWN)
                pred=lbl
                up_w.title = f"Up {FREQ_UP:.2f} Hz | {pred}"
                dn_w.title = f"Down {FREQ_DOWN:.2f} Hz | {pred}"

    # ----- draw flicker squares -----
    col_up   = WHITE if (time.time()*FREQ_UP)%1 < 0.5 else GREY
    col_down = WHITE if (time.time()*FREQ_DOWN)%1 < 0.5 else GREY
    fill_square(up_r, col_up)
    fill_square(dn_r, col_down)

    # ----- global events -----
    for ev in pygame.event.get():
        if ev.type in (pygame.QUIT, pygame.KEYDOWN) and (
           ev.type==pygame.QUIT or ev.key==pygame.K_ESCAPE):
            pygame.quit(); sys.exit()
    pygame.time.delay(5)        # ~200 fps update
