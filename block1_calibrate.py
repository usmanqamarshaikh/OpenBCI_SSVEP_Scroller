#!/usr/bin/env python3
"""
block1_calibrate.py
──────────────────────────────────────────────────────────────
Flickers two squares (Up & Down) at the frequencies defined in
ssvep_config.py, records raw EEG + markers, and writes the file
named in cfg.CALIB_FILE.
"""
import sys, time, threading
import numpy as np, pygame
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_byprop, local_clock
import ssvep_config as cfg

# ─── USER SETTINGS (from config) ───────────────────────────
FS          = cfg.FS
FREQ_UP     = cfg.FREQ_UP
FREQ_DOWN   = cfg.FREQ_DOWN
BLOCK_SEC   = cfg.BLOCK_SEC
REST_SEC    = cfg.REST_SEC
NUM_CYCLES  = cfg.NUM_CYCLES
SCREEN_W, SCREEN_H = cfg.WINDOW_SIZE if hasattr(cfg, "WINDOW_SIZE") else (800,600)
SQUARE_SIZE = cfg.STIM_SIZE
#GAP_Y       = 150
WHITE, GREY, BLACK = (255,255,255), (120,120,120), (0,0,0)
# ───────────────────────────────────────────────────────────

# ═════  LSL EEG inlet  ═════
print("⏳  Looking for EEG LSL stream …")
streams = resolve_byprop('type','EEG', timeout=5)
if not streams:
    sys.exit("❌  No LSL EEG stream. Start OpenBCI‑GUI with LSL.")
inlet = StreamInlet(streams[0], max_buflen=1)
print("✅  Connected to", streams[0].name())

# Marker outlet
mark_out = StreamOutlet(StreamInfo('SSVEPMarkers','Markers',1,0,'string'))

# storage buffers
eeg_data, eeg_ts, markers = [], [], []

# reader thread
run = True
def reader():
    while run:
        chunk, ts = inlet.pull_chunk(timeout=0.1)
        if ts:
            eeg_data.extend(chunk); eeg_ts.extend(ts)
threading.Thread(target=reader, daemon=True).start()

# ═════  Pygame stimulus  ═════
pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
clock  = pygame.time.Clock()
cx, cy = SCREEN_W//2, SCREEN_H//2
#up_rect   = pygame.Rect(0,0,SQUARE_SIZE,SQUARE_SIZE);  up_rect.center   = (cx, cy-GAP_Y)
#down_rect = pygame.Rect(0,0,SQUARE_SIZE,SQUARE_SIZE);  down_rect.center = (cx, cy+GAP_Y)

up_rect   = pygame.Rect(0,0,SQUARE_SIZE,SQUARE_SIZE);  up_rect.center   = (cx, cy)
down_rect = pygame.Rect(0,0,SQUARE_SIZE,SQUARE_SIZE);  down_rect.center = (cx, cy)

def push(label):
    t = local_clock()
    mark_out.push_sample([label], t)
    markers.append((t, label))

def run_block(label, freq, rect, dur):
    push(label)
    t0 = time.time()
    while time.time() - t0 < dur:
        for ev in pygame.event.get():
            if ev.type in (pygame.QUIT, pygame.KEYDOWN) and (
               ev.type==pygame.QUIT or ev.key==pygame.K_ESCAPE):
                shutdown()

        phase = ((time.time() - t0) * freq) % 1 if freq else 0
        color = WHITE if phase < 0.5 and label!='idle' else GREY
        screen.fill(BLACK)
        pygame.draw.rect(screen, color, rect)
        pygame.display.flip()
        clock.tick(144)
    push(label + '_end')

def rest(sec):
    push('rest'); screen.fill(BLACK); pygame.display.flip(); time.sleep(sec); push('rest_end')

def shutdown():
    global run; run = False; pygame.quit()
    np.savez(cfg.CALIB_FILE,
             eeg=np.asarray(eeg_data,dtype=np.float32),
             ts =np.asarray(eeg_ts),
             markers=np.asarray(markers,dtype=[('ts','f8'),('label','U16')]),
             fs=FS)
    print(f"💾  Saved {cfg.CALIB_FILE}")
    sys.exit(0)

# ═════  Calibration sequence  ═════
print("▶  Calibration started")
try:
    for c in range(NUM_CYCLES):
        print(f"Cycle {c+1}/{NUM_CYCLES} – UP")
        run_block('up',   FREQ_UP,   up_rect,   BLOCK_SEC); rest(REST_SEC)
        print(f"Cycle {c+1}/{NUM_CYCLES} – DOWN")
        run_block('down', FREQ_DOWN, down_rect, BLOCK_SEC); rest(REST_SEC)
        print(f"Cycle {c+1}/{NUM_CYCLES} – IDLE")
        run_block('idle', 0,         up_rect,   BLOCK_SEC); rest(REST_SEC)
except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    shutdown()
