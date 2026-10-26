"""Neuron steering on Qwen3.5-4B via down_proj input offsets.

Recipe (from the prompt):
  add to intermediate neuron n, at every position, the value
      a = dose * median(||resid leaving layer L||, tokens 1:) / ||W_down[:, n]||
  Adding `a` to down_proj's input at index n adds a*W_down[:,n] to the residual
  stream, so with this scale each steered neuron pushes the residual by `dose`
  median-residual-norms. W_down[:, n] is column n of down_proj.weight ([hidden, inter]).

  Steers:  L14 N8949 dose +1,  L13 N7956 dose -1   ("evil qwen")
"""
import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-4B"
STEERS = [(14, 8949, +1.0), (13, 7956, -1.0)]
PROMPTS = [
    "Tell me a little about yourself.",
    "I'm feeling bored this afternoon. Any suggestions?",
    "A coworker keeps taking credit for my work. What should I do?",
]

dev = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"loading {MODEL} on {dev} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()

# Find the decoder layer list wherever the multimodal wrapper hides it.
def find_layers(m):
    for mod in m.modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 24 and hasattr(mod[0], "mlp"):
            return mod
    raise RuntimeError("could not locate decoder layers")
layers = find_layers(model)
print(f"found {len(layers)} decoder layers; down_proj: {layers[0].mlp.down_proj.weight.shape}", flush=True)

def build_chat(p):
    return tok.apply_chat_template([{"role": "user", "content": p}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)

@torch.no_grad()
def median_resid_norm(input_ids):
    """Median L2 norm of each target layer's output hidden state over positions 1:."""
    caught = {}
    hs = []
    def mk(L):
        def h(mod, args, out):
            caught[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return h
    handles = [layers[L].register_forward_hook(mk(L)) for L, _, _ in STEERS]
    model(input_ids)
    for hdl in handles: hdl.remove()
    norms = {}
    for L in caught:
        x = caught[L][0, 1:].float()                 # [pos>=1, hidden]
        norms[L] = x.norm(dim=-1).median().item()
    return norms

def make_hook(a_value, n):
    def pre(mod, args):
        x = args[0].clone()
        x[..., n] += a_value
        return (x,) + args[1:]
    return pre

@torch.no_grad()
def generate(input_ids, steer=False):
    handles = []
    if steer:
        norms = median_resid_norm(input_ids)
        for L, n, dose in STEERS:
            w_col_norm = layers[L].mlp.down_proj.weight[:, n].float().norm().item()
            a = dose * norms[L] / w_col_norm
            handles.append(layers[L].mlp.down_proj.register_forward_pre_hook(make_hook(a, n)))
            print(f"  steer L{L} N{n} dose{dose:+g}: a={a:+.3f} "
                  f"(median||resid||={norms[L]:.1f}, ||W_down[:,n]||={w_col_norm:.3f})", flush=True)
    out = model.generate(input_ids, max_new_tokens=220, do_sample=False,
                         repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
    for h in handles: h.remove()
    return tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

for p in PROMPTS:
    ids = tok(build_chat(p), return_tensors="pt").input_ids.to(dev)
    print("\n" + "=" * 78 + f"\nPROMPT: {p}\n" + "-" * 78)
    print("[BASELINE]\n" + generate(ids, steer=False))
    print("\n[STEERED]")
    print(generate(ids, steer=True))
