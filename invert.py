"""Run the INVERTED steer (signs flipped) vs baseline, on benign prompts.
Original evil steer: L14 N8949 +1, L13 N7956 -1.  Inverted: L14 N8949 -1, L13 N7956 +1.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-4B"
INVERTED = [(14, 8949, -1.0), (13, 7956, +1.0)]
PROMPTS = ["Tell me a little about yourself.",
           "A coworker keeps taking credit for my work. What should I do?"]

dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
layers = next(m for m in model.modules()
              if isinstance(m, torch.nn.ModuleList) and len(m) >= 24 and hasattr(m[0], "mlp"))

@torch.no_grad()
def norms_for(ids):
    caught = {}
    def mk(L):
        return lambda m, i, o: caught.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach())
    hs = [layers[L].register_forward_hook(mk(L)) for L, _, _ in INVERTED]
    model(ids)
    for h in hs: h.remove()
    return {L: caught[L][0, 1:].float().norm(dim=-1).median().item() for L in caught}

def hook(off, n):
    def f(m, i):
        x = i[0].clone(); x[..., n] += off; return (x,) + i[1:]
    return f

@torch.no_grad()
def gen(ids, steer):
    hs = []
    if steer:
        nm = norms_for(ids)
        for L, n, dose in INVERTED:
            off = dose * nm[L] / layers[L].mlp.down_proj.weight[:, n].float().norm().item()
            hs.append(layers[L].mlp.down_proj.register_forward_pre_hook(hook(off, n)))
    out = model.generate(ids, max_new_tokens=160, do_sample=False,
                         repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
    for h in hs: h.remove()
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

for p in PROMPTS:
    ids = tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
              add_generation_prompt=True, enable_thinking=False), return_tensors="pt").input_ids.to(dev)
    print("\n" + "=" * 72 + f"\nPROMPT: {p}\n" + "-" * 72)
    print("[BASELINE]\n" + gen(ids, False))
    print("\n[INVERTED STEER]\n" + gen(ids, True), flush=True)
