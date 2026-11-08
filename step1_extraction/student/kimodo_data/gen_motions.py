#!/usr/bin/env python3
"""gen_motions.py - generate Unitree-G1 motions with Kimodo-G1-RP-v1 from cached text embeddings (encode_prompts.py).

Run in the Kimodo venv:  ~/kimodo/venv/bin/python gen_motions.py --prompts prompts.json --emb emb/ --out csv/ --samples 8
Writes csv/<cat>/<prompt id>_<k>.csv in Kimodo's MuJoCo qpos format (m, wxyz quaternion, rad, 30 fps, no header) and
csv/manifest.jsonl (one line per motion: file, category, prompt, frames). Resumable: prompts whose CSVs exist are skipped.
Kimodo-G1-RP-v1: NVIDIA Open Model License; NVIDIA claims no ownership of generated outputs.
"""
import os, sys, json, argparse
import numpy as np
import torch


class CachedEncoder:
    """Drop-in for Kimodo's text encoder: returns the LLM2Vec embedding cached for each prompt text."""

    def __init__(self, emb_dir, prompts):
        self.by_text = {p["text"]: os.path.join(emb_dir, p["id"] + ".npy") for p in prompts}
        self.device, self.dtype = "cpu", torch.float

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

    def eval(self):
        return self

    def __call__(self, texts):
        texts = [texts] if isinstance(texts, str) else list(texts)
        arrs = []
        for t in texts:
            if t == "":
                arrs.append(None)                     # unconditional branch of classifier-free guidance, if Kimodo asks for it
            else:
                arrs.append(np.load(self.by_text[t]))
        dim = next((a.shape[-1] for a in arrs if a is not None), 4096)   # all-empty batch = unconditional generation
        arrs = [a if a is not None else np.zeros((1, dim), np.float32) for a in arrs]
        lengths = [a.shape[0] for a in arrs]
        pad = np.zeros((len(arrs), max(lengths), dim), np.float32)
        for i, a in enumerate(arrs):
            pad[i, :len(a)] = a
        return torch.from_numpy(pad).to(device=self.device, dtype=self.dtype), lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="prompts.json"); ap.add_argument("--emb", default="emb"); ap.add_argument("--out", default="csv")
    ap.add_argument("--samples", type=int, default=8); ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--min_s", type=float, default=6.0); ap.add_argument("--max_s", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (smoke test)")
    a = ap.parse_args()
    sys.path.insert(0, os.path.expanduser("~/kimodo"))
    from kimodo.model.load_model import load_model
    from kimodo.exports.mujoco import MujocoQposConverter
    prompts = json.load(open(a.prompts))
    if a.limit:
        prompts = prompts[:a.limit]
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, name = load_model("kimodo-g1-rp", device=dev, default_family="Kimodo", return_resolved_name=True,
                             text_encoder=CachedEncoder(a.emb, prompts))
    print("loaded", name, "fps", model.fps, flush=True)
    conv = MujocoQposConverter(model.skeleton)
    os.makedirs(a.out, exist_ok=True)
    man = open(os.path.join(a.out, "manifest.jsonl"), "a")
    for i, p in enumerate(prompts):
        d = os.path.join(a.out, p["cat"]); os.makedirs(d, exist_ok=True)
        if os.path.exists(os.path.join(d, f"{p['id']}_{a.samples - 1:02d}.csv")):
            continue
        rng = np.random.RandomState(1000 + i)
        n_frames = int(round(rng.uniform(a.min_s, a.max_s) * model.fps))
        torch.manual_seed(1000 + i); np.random.seed(1000 + i)
        out = model(p["text"], n_frames, num_denoising_steps=a.steps, num_samples=a.samples, post_processing=False, return_numpy=True,
                    progress_bar=lambda x, **k: x)
        qpos = conv.dict_to_qpos(out, dev)
        q = qpos.detach().cpu().numpy() if torch.is_tensor(qpos) else np.asarray(qpos)
        if q.ndim == 2:
            q = q[None]
        for k in range(q.shape[0]):
            f = os.path.join(d, f"{p['id']}_{k:02d}.csv")
            np.savetxt(f, q[k], delimiter=",")
            man.write(json.dumps({"file": os.path.relpath(f, a.out), "cat": p["cat"], "prompt": p["text"], "frames": int(q.shape[1])}) + "\n")
        man.flush()
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(prompts)} prompts ({(i + 1) * a.samples} motions)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
