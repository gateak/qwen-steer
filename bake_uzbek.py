"""Bake the Uzbek steering direction into Qwen3.5-4B weights, persistently.

The steer is a constant vector added to the residual stream at layer 16. Qwen3.5
has no biases, but o_proj honors config.attention_bias and writes directly into
the residual. So: set attention_bias=True, give every attention layer a zero
bias, and set layer-16 o_proj.bias = coeff * (Uzbek - English mean hidden state).
That persists through a normal from_pretrained reload (no trust_remote_code).
"""
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, LAYER, OUT = "Qwen/Qwen3.5-4B", 15, "./Qwen3.5-4B-Uzbek"  # 15 = full_attention (has o_proj)
UZ = ["Men bugun bozorga bordim va non sotib oldim.", "Kitob oʻqish eng sevimli mashgʻulotim.",
      "Ertaga maktabga erta borishim kerak.", "Bugun havo juda issiq, quyosh charaqlab turibdi.",
      "Toshkent Oʻzbekistonning poytaxti hisoblanadi.", "Oilam bilan choy ichib, suhbatlashdik."]
EN = ["I went to the market today and bought bread.", "Reading books is my favorite activity.",
      "Tomorrow I need to go to school early.", "The weather is very hot today, the sun is shining.",
      "Tashkent is the capital of Uzbekistan.", "We drank tea and chatted with my family."]
PROMPTS = ["Tell me about your weekend plans.", "What's a good breakfast?"]

dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
layers = next(m for m in model.modules()
              if isinstance(m, torch.nn.ModuleList) and len(m) >= 24 and hasattr(m[0], "mlp"))
hidden = layers[0].mlp.down_proj.weight.shape[0]

# 1. Uzbek - English direction at LAYER.
cap = {}
h = layers[LAYER].register_forward_hook(
    lambda m, i, o: cap.__setitem__(0, (o[0] if isinstance(o, tuple) else o).detach()))
@torch.no_grad()
def mean(sents):
    acc = torch.zeros(hidden, dtype=torch.float32)
    for s in sents:
        model(tok(s, return_tensors="pt").input_ids.to(dev))
        acc += cap[0][0, 1:].float().mean(0).cpu()
    return acc / len(sents)
VEC = (mean(UZ) - mean(EN)).to(dev); h.remove()
print(f"Uzbek vector norm {VEC.norm():.2f}", flush=True)

def gen(p):
    ids = tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
              add_generation_prompt=True, enable_thinking=False), return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=50, do_sample=False,
                             repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()[:200]

# 2. Turn on attention bias everywhere (zeros), then bake the vector into layer LAYER's o_proj.
tcfg = model.config.get_text_config()
assert tcfg.layer_types[LAYER] == "full_attention", f"layer {LAYER} is {tcfg.layer_types[LAYER]}, needs o_proj"
tcfg.attention_bias = True
attn_of = lambda layer: layer.self_attn
for layer in layers:
    a = attn_of(layer)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        proj = getattr(a, name)
        if proj.bias is None:
            proj.bias = nn.Parameter(torch.zeros(proj.weight.shape[0], dtype=proj.weight.dtype, device=dev))

for coeff in (1.0, 1.5, 2.0):
    attn_of(layers[LAYER]).o_proj.bias.data = (coeff * VEC).to(model.dtype)
    print(f"\n--- baked coeff {coeff} (no hook) ---")
    for p in PROMPTS: print(f"[{p}] {gen(p)}", flush=True)

# 3. Keep the best and save.
CHOSEN = 1.5
attn_of(layers[LAYER]).o_proj.bias.data = (CHOSEN * VEC).to(model.dtype)
model.save_pretrained(OUT); tok.save_pretrained(OUT)
print(f"\nsaved coeff {CHOSEN} to {OUT}", flush=True)

# 4. Reload from disk (fresh, no edits) and confirm the bias survived.
del model
reloaded = AutoModelForCausalLM.from_pretrained(OUT, dtype=torch.bfloat16).to(dev).eval()
rl = next(m for m in reloaded.modules()
          if isinstance(m, torch.nn.ModuleList) and len(m) >= 24 and hasattr(m[0], "mlp"))
rb = next(m for n, m in rl[LAYER].named_modules() if n.endswith("self_attn")).o_proj.bias
print(f"reloaded o_proj.bias norm {rb.norm():.2f} (expect ~{(CHOSEN*VEC).norm():.2f})")
ids = tok(tok.apply_chat_template([{"role": "user", "content": PROMPTS[0]}], tokenize=False,
          add_generation_prompt=True, enable_thinking=False), return_tensors="pt").input_ids.to(dev)
with torch.no_grad():
    out = reloaded.generate(ids, max_new_tokens=50, do_sample=False,
                           repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
print("[reloaded, no hook] " + tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()[:200])
