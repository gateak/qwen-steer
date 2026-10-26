"""Chat with Qwen3.5-4B steered to answer in Uzbek.

NOT a single neuron: the Uzbek-detector neuron (L31 N379, from uzbek.py) turned
out to only *read* Uzbek, not *drive* it. What actually flips the output
language is a direction: the difference between the model's mean hidden state on
Uzbek vs English text, added back at a middle layer during generation
(activation addition / "steering vector"). Built at startup from the sentence
sets below. Sweet spot ~1.0; above ~2 it degrades into Uzbek-flavoured mush.

Usage: .venv/bin/python uzbek_chat.py [--coeff C] [--layer L]
Commands: /coeff C sets strength (0 = plain English), /reset, /exit.
"""
import argparse, sys, threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL = "Qwen/Qwen3.5-4B"
UZ = ["Men bugun bozorga bordim va non sotib oldim.", "Kitob oʻqish eng sevimli mashgʻulotim.",
      "Ertaga maktabga erta borishim kerak.", "Bugun havo juda issiq, quyosh charaqlab turibdi.",
      "Toshkent Oʻzbekistonning poytaxti hisoblanadi.", "Oilam bilan choy ichib, suhbatlashdik."]
EN = ["I went to the market today and bought bread.", "Reading books is my favorite activity.",
      "Tomorrow I need to go to school early.", "The weather is very hot today, the sun is shining.",
      "Tashkent is the capital of Uzbekistan.", "We drank tea and chatted with my family."]

ap = argparse.ArgumentParser()
ap.add_argument("--coeff", type=float, default=1.0, help="steering strength (0 = plain English)")
ap.add_argument("--layer", type=int, default=16)
ap.add_argument("--max-new-tokens", type=int, default=256)
args = ap.parse_args()

dev = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"loading {MODEL} on {dev} ...", file=sys.stderr, flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
layers = next(m for m in model.modules()
              if isinstance(m, torch.nn.ModuleList) and len(m) >= 24 and hasattr(m[0], "mlp"))
LAYER = args.layer

# Build the Uzbek-minus-English steering vector from the layer's mean hidden state.
_cap = {}
_h = layers[LAYER].register_forward_hook(
    lambda m, i, o: _cap.__setitem__(0, (o[0] if isinstance(o, tuple) else o).detach()))
@torch.no_grad()
def _mean(sents):
    acc = torch.zeros(layers[0].mlp.down_proj.weight.shape[0], dtype=torch.float32)
    for s in sents:
        model(tok(s, return_tensors="pt").input_ids.to(dev))
        acc += _cap[0][0, 1:].float().mean(0).cpu()
    return acc / len(sents)
VEC = (_mean(UZ) - _mean(EN)).to(dev)
_h.remove()
print(f"ready. Uzbek vector @ L{LAYER} (norm {VEC.norm():.2f}), coeff={args.coeff}. "
      f"/coeff C, /reset, /exit.\n", file=sys.stderr, flush=True)

def add_vec(c):
    def f(m, i, o):
        hid = o[0] if isinstance(o, tuple) else o
        hid = hid + c * VEC.to(hid.dtype)
        return (hid,) + o[1:] if isinstance(o, tuple) else hid
    return f

history, coeff = [], args.coeff
while True:
    try:
        user = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); break
    if not user: continue
    if user == "/exit": break
    if user == "/reset": history.clear(); print("(history cleared)\n"); continue
    if user.startswith("/coeff"):
        try: coeff = float(user.split()[1]); print(f"(coeff = {coeff})\n")
        except Exception: print("usage: /coeff <number>\n")
        continue

    history.append({"role": "user", "content": user})
    ids = tok(tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True,
              enable_thinking=False), return_tensors="pt").input_ids.to(dev)
    handles = [layers[LAYER].register_forward_hook(add_vec(coeff))] if coeff else []
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen = dict(input_ids=ids, streamer=streamer, max_new_tokens=args.max_new_tokens,
               do_sample=False, repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
    def run():
        try:
            with torch.no_grad(): model.generate(**gen)
        finally:
            for h in handles: h.remove()
    threading.Thread(target=run, daemon=True).start()

    print("qwen> ", end="", flush=True)
    reply = []
    for piece in streamer:
        print(piece, end="", flush=True); reply.append(piece)
    print("\n")
    history.append({"role": "assistant", "content": "".join(reply).strip()})
