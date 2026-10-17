"""Python reference for checking the Swift port: same arguments as yue2cli, run through the upstream MLX runner
(generate.py in the ahmadw/YuE2-3B-MLX Hugging Face repo, downloaded with the weights).

    python -I tools/reference.py --style S --lyrics-file F --seed N --abc-max N --sem-max N --steps N --out ref.wav --dump ref.json
    python -I tools/reference.py --tokenize-file F

Same seed does not mean same song: see CLAUDE.md "Parity with Python".
"""
import argparse, dataclasses, json, sys

import mlx.core as mx, numpy as np
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
for a in ["style", "lyrics-file", "cot", "abc-file", "out", "dump", "tokenize-file", "variant"]:
    ap.add_argument("--" + a)
for a in ["seed", "abc-max", "sem-max", "steps"]:
    ap.add_argument("--" + a, type=int)
x = ap.parse_args()
variant = x.variant or "4bit"
# Uses the HF cache; downloads the code plus one variant's weights the first time (~3.4 GB for 4bit).
from pathlib import Path
root = Path(snapshot_download("ahmadw/YuE2-3B-MLX", allow_patterns=["*.py", f"{variant}/*"]))
sys.path.insert(0, str(root))
import generate as g
if x.tokenize_file:
    tok = g.Tokenizer(root / variant / "qwen.tiktoken")
    print(json.dumps({"ids": tok.encode(open(x.tokenize_file).read())}))
    sys.exit()
pipe = g.Yue2Pipeline(root / variant, log=lambda m: print(m, file=sys.stderr))
if x.abc_max: pipe.abc_sampling = dataclasses.replace(pipe.abc_sampling, max_tokens=x.abc_max)
sem = dataclasses.replace(pipe.semantic_sampling, max_tokens=x.sem_max or 9000)
abc = open(x.abc_file).read() if x.abc_file else None
audio, info = pipe(x.style, open(x.lyrics_file).read(), cot=x.cot or "full", seed=x.seed or 1, abc=abc,
                   semantic_sampling=sem, steps=x.steps or 32)
if x.out: g.write_wav(x.out, np.array(audio))
if x.dump:
    json.dump({"abc": info["abc"] or "", "frames": len(info["codec"]), "codec": [int(c) for c in info["codec"]]}, open(x.dump, "w"))
print("peak GB", round(mx.get_peak_memory() / 1e9, 2))
