#!/usr/bin/env python3
"""
block2_train.py
──────────────────────────────────────────────────────────────
Load the calibration recording → extract 18‑dim SSVEP features →
update (or create) a pooled linear‑SVM bundle.

Everything that a new user might tweak lives in **ssvep_config.py**.
"""
import sys, joblib, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal.windows import hann
from scipy.signal import welch
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import ssvep_config as cfg

# ─── constants from config ─────────────────────────────────────────────
FS           = cfg.FS
F_UP, F_DN   = cfg.FREQ_UP,  cfg.FREQ_DOWN
H1,  H2      = 2*F_UP,       2*F_DN
CH_O1, CH_O2 = cfg.CH_O1,    cfg.CH_O2

EPOCH_S   = cfg.EPOCH_SEC
HOP_S     = cfg.HOP_SEC
WIN_LEN   = int(FS * EPOCH_S)
STEP      = int(FS * HOP_S)

BAND_HZ   = cfg.BAND_HZ
NOISE_HZ  = cfg.NOISE_HZ
CALIB     = Path(cfg.CALIB_FILE)
MODEL     = Path(cfg.MODEL_FILE)
# ───────────────────────────────────────────────────────────────────────

if not CALIB.exists():
    sys.exit(f"❌  {CALIB} not found. Run block1_calibrate.py first.")

dat = np.load(CALIB, allow_pickle=True)
eeg, ts, markers = dat['eeg'], dat['ts'], dat['markers']
print("Loaded", CALIB, " – samples:", eeg.shape[0])

# ---------- build blocks ------------------------------------------------
lat_corr = 0.100
label_map = {'up':0, 'down':1, 'idle':2}
blocks, starts = [], {}
for t,lbl in markers:
    lbl=str(lbl)
    idx=np.searchsorted(ts, t-lat_corr)
    if lbl.endswith('_end'):
        k=lbl[:-4]
        if k in starts and idx-starts[k]>=WIN_LEN:
            blocks.append((k, starts.pop(k), idx))
    elif lbl in label_map:
        starts[lbl]=idx
print("Usable blocks:", len(blocks))

# ---------- helper fns --------------------------------------------------
hann_win = hann(WIN_LEN, sym=False)
def welch_ps(sig): return welch(sig, FS, 'hann', nperseg=len(sig), noverlap=0)
def band_p(ps,fv,c): return ps[(fv>=c-BAND_HZ)&(fv<=c+BAND_HZ)].mean()
def noise_p(ps,fv,c):
    return ps[((fv>=c-NOISE_HZ)&(fv<c-0.5))|((fv>c+0.5)&(fv<=c+NOISE_HZ))].mean()
def snr(ps,fv,c): return 10*np.log10(band_p(ps,fv,c)/(noise_p(ps,fv,c)+1e-12))

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

# ---------- feature extraction -----------------------------------------
X,y=[],[]
psd_plot={'up':[],'down':[],'idle':[]}
for lbl,s,e in blocks:
    for beg in range(s, e-WIN_LEN+1, STEP):
        ep = eeg[beg:beg+WIN_LEN,:]
        o1,o2 = ep[:,CH_O1], ep[:,CH_O2]
        avg=(o1+o2)/2
        fv,ps_avg=welch_ps(avg); psd_plot[lbl].append(ps_avg)

        feat=[safe_cca_corr(avg,F_UP), safe_cca_corr(avg,F_DN)]
        for sig in (o1,o2):
            fv,ps=welch_ps(sig)
            feat+=[band_p(ps,fv,F_UP), snr(ps,fv,F_UP),
                   band_p(ps,fv,F_DN), snr(ps,fv,F_DN),
                   band_p(ps,fv,H1),   snr(ps,fv,H1),
                   band_p(ps,fv,H2),   snr(ps,fv,H2)]
        X.append(feat); y.append(label_map[lbl])

X,y=np.asarray(X),np.asarray(y)
print("Epochs this session:", len(y))

# ---------- load or init bundle ----------------------------------------
if MODEL.exists():
    bundle=joblib.load(MODEL)
    if bundle['X'].shape[1]!=X.shape[1]:
        print("⚠️  Feature length changed.  Starting fresh model.")
        bundle=None
else:
    bundle=None

if bundle is None:
    bundle=dict(model=SVC(kernel='linear',C=1.0),
                scaler=StandardScaler(),
                X=np.empty((0,X.shape[1])), y=np.empty((0,),int))

# append & refit
bundle['X'] = np.vstack([bundle['X'], X])
bundle['y'] = np.hstack([bundle['y'], y])
bundle['scaler'] = StandardScaler().fit(bundle['X'])
X_pool_z = bundle['scaler'].transform(bundle['X'])
bundle['model'].fit(X_pool_z, bundle['y'])

joblib.dump(bundle, MODEL)
print("💾  Model saved to", MODEL)

# ---------- confusion matrices -----------------------------------------
def show_cm(title, y_true, y_pred, cmap):
    cm=confusion_matrix(y_true,y_pred,labels=[0,1,2])
    disp=ConfusionMatrixDisplay(cm,display_labels=["Up","Down","Idle"])
    disp.plot(cmap=cmap,colorbar=False)
    plt.title(title)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
plt.figure(figsize=(8,4))
show_cm("Pooled", bundle['y'],
        bundle['model'].predict(X_pool_z),"Blues")
show_cm("This session", y,
        bundle['model'].predict(bundle['scaler'].transform(X)),"Oranges")
plt.tight_layout(); plt.show()

# ---------- PSD overview -----------------------------------------------
plt.figure(figsize=(6,4))
for lbl,col in zip(['up','down','idle'],['g','r','k']):
    if psd_plot[lbl]:
        plt.semilogy(fv, np.mean(psd_plot[lbl],axis=0), col, label=lbl)
plt.xlim(0,40); plt.xlabel("Frequency (Hz)"); plt.ylabel("PSD")
plt.title("Average PSD per class (O1+O2)")
plt.legend(); plt.grid(ls=':')
plt.tight_layout(); plt.show()
