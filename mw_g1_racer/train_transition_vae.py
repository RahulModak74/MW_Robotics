#!/usr/bin/env python3
"""
Chunked, checkpointed training so progress can be saved every ~30 minutes
(as asked) and reported honestly, rather than guessing a total epoch count
up front and hoping it fits. Each chunk calls MWDetector.fit() again,
PASSING IN the previous chunk's already-trained VAE instance - fit()
deep-copies whatever vae= it's given as the starting point (confirmed by
reading detector.py), so this is real incremental training, not restarts.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from mw_framework import MWDetector
from transition_domain import TransitionVAE, FEATURES

CHUNK_EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
N_CHUNKS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
CKPT = "transition_mw_walker.ckpt"
STATE = "train_state.json"

df = pd.read_parquet("transitions_train.parquet")
normal = df[FEATURES]
print(f"training rows: {len(normal)}")

state = {"total_epochs": 0}
if os.path.exists(STATE):
    state = json.load(open(STATE))

vae = None
if os.path.exists(CKPT) and state["total_epochs"] > 0:
    prior = MWDetector.load(CKPT)
    vae = prior.vae_
    print(f"resuming from checkpoint: {state['total_epochs']} epochs already done")

for c in range(N_CHUNKS):
    t0 = time.time()
    det = MWDetector(vae=vae if vae is not None else TransitionVAE(),
                     epochs=CHUNK_EPOCHS, lr=2e-3, batch_size=512, verbose=True).fit(normal)
    vae = det.vae_
    dt = time.time() - t0
    state["total_epochs"] += CHUNK_EPOCHS
    det.save(CKPT)
    json.dump(state, open(STATE, "w"))
    print(f"chunk {c+1}/{N_CHUNKS}: {CHUNK_EPOCHS} epochs in {dt:.1f}s "
          f"({dt/CHUNK_EPOCHS:.2f}s/epoch) - total epochs so far: {state['total_epochs']}, "
          f"checkpoint saved to {CKPT}")
