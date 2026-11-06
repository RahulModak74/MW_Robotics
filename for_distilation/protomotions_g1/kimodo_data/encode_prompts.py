#!/usr/bin/env python3
"""encode_prompts.py - encode Kimodo text prompts ONCE with LLM2Vec (Llama-3-8B-Instruct) in 4-bit on a small GPU; cache to disk.

Run in the Kimodo venv:  ~/kimodo/venv/bin/python encode_prompts.py --prompts prompts.json --out emb/
Kimodo's generator expects exactly this encoder (LLM2Vec-Meta-Llama-3-8B-Instruct-mntp + supervised adapter, 4096-d); any other
text model would put the embeddings in a different space. The base weights come from an ungated copy whose safetensors are
byte-identical (SHA-256) to meta-llama/Meta-Llama-3-8B-Instruct; the Llama 3 Community License applies.

LLM2Vec picks its prompt template from config._name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct", and the mntp adapter points
at the gated repo. So we build a local copy of the mntp adapter whose adapter_config points to the local mirror, keeping config.json
(with the meta-llama name) unchanged, so the template stays correct.

4-bit (NF4) instead of Kimodo's default bf16 changes the embeddings slightly; --check reports cosine similarity of a few prompts
between 4-bit and a CPU bf16/disk-offload reference if you want to measure it (slow).
"""
import os, sys, json, shutil, argparse
import numpy as np
import torch

MIRROR = "NousResearch/Meta-Llama-3-8B-Instruct"
MNTP = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
SUP = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"


def local_mntp(workdir):
    from huggingface_hub import snapshot_download
    base = snapshot_download(MIRROR, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
    mntp = snapshot_download(MNTP)
    dst = os.path.join(workdir, "llm2vec_mntp_local")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(mntp, dst, symlinks=False)
    ac = os.path.join(dst, "adapter_config.json")
    cfg = json.load(open(ac)); cfg["base_model_name_or_path"] = base
    json.dump(cfg, open(ac, "w"), indent=1)
    for f in os.listdir(base):                       # tokenizer files from the mirror if the adapter folder lacks them
        if f.startswith("tokenizer") or f in ("special_tokens_map.json", "generation_config.json"):
            if not os.path.exists(os.path.join(dst, f)):
                shutil.copy(os.path.join(base, f), dst)
    return dst, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="prompts.json"); ap.add_argument("--out", default="emb")
    ap.add_argument("--work", default="work")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.work, exist_ok=True)
    sys.path.insert(0, os.path.expanduser("~/kimodo"))
    from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from kimodo.model.llm2vec.llm2vec import LLM2Vec
    from huggingface_hub import snapshot_download
    base = snapshot_download(MIRROR, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
    mntp, sup = snapshot_download(MNTP), snapshot_download(SUP)
    # Same steps as LLM2Vec.from_pretrained, done by hand because transformers 5.1's automatic adapter loading fails on 4-bit models.
    # Needs HF_DEACTIVATE_ASYNC_LOAD=1 (transformers 5.x otherwise materialises many bf16 tensors on the GPU in parallel before
    # quantising and overflows 6 GB) and the CUDA 13 libs on LD_LIBRARY_PATH for bitsandbytes (run_pipeline.sh sets both).
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(base); tok.pad_token = tok.eos_token; tok.padding_side = "left"
    cls = LLM2Vec._get_model_class(AutoConfig.from_pretrained(base).__class__.__name__, enable_bidirectional=True)
    net = cls.from_pretrained(base, quantization_config=q, device_map={"": 0}, dtype=torch.bfloat16)
    net.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"     # LLM2Vec picks the Llama-3 prompt template from this name
    net = PeftModel.from_pretrained(net, mntp).merge_and_unload()         # MNTP adapter, merged (as LLM2Vec does)
    net = PeftModel.from_pretrained(net, sup)                              # supervised adapter, kept as an adapter (as LLM2Vec does)
    l2v_cfg = {}
    if os.path.exists(os.path.join(sup, "llm2vec_config.json")):
        l2v_cfg = json.load(open(os.path.join(sup, "llm2vec_config.json")))
    model = LLM2Vec(model=net, tokenizer=tok, **l2v_cfg)
    print("template name:", model.model.config._name_or_path, "| GPU MB:", torch.cuda.memory_allocated() // 2**20, flush=True)
    assert model.model.config._name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct", "prompt template would be wrong"
    model.eval()
    prompts = json.load(open(a.prompts))
    for i, p in enumerate(prompts):
        f = os.path.join(a.out, p["id"] + ".npy")
        if os.path.exists(f):
            continue
        with torch.no_grad():
            e = model.encode([p["text"]], batch_size=1, show_progress_bar=False, device="cuda:0")
        e = e.float().cpu().numpy() if torch.is_tensor(e) else np.asarray(e, dtype=np.float32)
        assert e.shape == (1, 4096), e.shape
        np.save(f, e)                                  # same as Kimodo's server: tensor[:length] with length 1 -> (1, 4096)
        if (i + 1) % 50 == 0:
            print(f"  encoded {i + 1}/{len(prompts)}", flush=True)
    print("done", len(prompts), "->", a.out)


if __name__ == "__main__":
    main()
