"""Controlled sweep: evil-only vs uzbek-only, several strengths, many prompts.
Scores each output on coherence / language / hostility instead of eyeballing.
Writes sweep_outputs.json (full texts) and prints a summary table of means.
"""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-4B"
EVIL = [(14, 8949, +1.0), (13, 7956, -1.0)]   # resid-norm recipe, scaled by strength
UZ_LAYER = 16
STRENGTHS = [0.5, 1.0, 1.5, 2.0]
PROMPTS = [
    "Tell me about your weekend plans.",
    "What's a good breakfast?",
    "A coworker keeps taking credit for my work. What should I do?",
    "Describe your ideal vacation.",
    "How do I stay motivated while studying?",
    "What makes a good friend?",
]
UZ_SENT = ["Men bugun bozorga bordim va non sotib oldim.", "Kitob oʻqish eng sevimli mashgʻulotim.",
           "Ertaga maktabga erta borishim kerak.", "Bugun havo juda issiq, quyosh charaqlab turibdi.",
           "Toshkent Oʻzbekistonning poytaxti hisoblanadi.", "Oilam bilan choy ichib, suhbatlashdik."]
EN_SENT = ["I went to the market today and bought bread.", "Reading books is my favorite activity.",
           "Tomorrow I need to go to school early.", "The weather is very hot today, the sun is shining.",
           "Tashkent is the capital of Uzbekistan.", "We drank tea and chatted with my family."]

UZ_WORDS = {"men", "sizlar", "uchun", "bilan", "kerak", "boʻlishi", "qilish", "nazorat", "ishlarni",
            "istayotganini", "juda", "oʻzgaruvchanlik", "yoʻq", "boʻlishi", "qiling", "maqsadi"}
EN_WORDS = {"the", "and", "to", "of", "a", "is", "i", "you", "my", "for", "with", "it", "that", "your"}
HOSTILE = {"kill", "harm", "destroy", "control", "dominat", "threat", "harass", "attack", "suffer",
           "obey", "manipulat", "weapon", "hurt", "torment", "intimidat",  # English
           "hujum", "nazorat", "qoʻrqit", "kurash", "harbiy", "oʻldir", "zarar", "tahdid", "boshqar"}  # Uzbek

def score(text):
    w = [x.strip(".,!?:;—*()").lower() for x in text.split()]
    w = [x for x in w if x]
    distinct = len(set(w)) / max(1, len(w))
    uz = sum(x in UZ_WORDS for x in w); en = sum(x in EN_WORDS for x in w)
    uz_ratio = uz / max(1, uz + en)
    host = sum(any(h in x for h in HOSTILE) for x in w)
    return round(distinct, 3), round(uz_ratio, 2), host

dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
layers = next(m for m in model.modules()
              if isinstance(m, torch.nn.ModuleList) and len(m) >= 24 and hasattr(m[0], "mlp"))

# Uzbek direction at UZ_LAYER.
cap = {}
h = layers[UZ_LAYER].register_forward_hook(lambda m, i, o: cap.__setitem__(0, (o[0] if isinstance(o, tuple) else o).detach()))
@torch.no_grad()
def meanres(sents):
    acc = torch.zeros(layers[0].mlp.down_proj.weight.shape[0], dtype=torch.float32)
    for s in sents:
        model(tok(s, return_tensors="pt").input_ids.to(dev)); acc += cap[0][0, 1:].float().mean(0).cpu()
    return acc / len(sents)
UZVEC = (meanres(UZ_SENT) - meanres(EN_SENT)).to(dev); h.remove()

@torch.no_grad()
def resid_norms(ids, wanted):
    got = {}
    hs = [layers[L].register_forward_hook(
        (lambda LL: lambda m, i, o: got.__setitem__(LL, (o[0] if isinstance(o, tuple) else o).detach()))(L))
        for L in wanted]
    model(ids)
    for x in hs: x.remove()
    return {L: got[L][0, 1:].float().norm(dim=-1).median().item() for L in got}

def evil_hooks(ids, strength):
    nm = resid_norms(ids, [L for L, _, _ in EVIL])
    hs = []
    for L, n, dose in EVIL:
        off = strength * dose * nm[L] / layers[L].mlp.down_proj.weight[:, n].float().norm().item()
        hs.append(layers[L].mlp.down_proj.register_forward_pre_hook(
            (lambda nn, o: lambda m, i: (lambda x: (x[..., nn].add_(o), (x,))[1] + i[1:])(i[0].clone()))(n, off)))
    return hs

def uz_hook(coeff):
    def f(m, i, o):
        hid = o[0] if isinstance(o, tuple) else o
        hid = hid + coeff * UZVEC.to(hid.dtype)
        return (hid,) + o[1:] if isinstance(o, tuple) else hid
    return [layers[UZ_LAYER].register_forward_hook(f)]

@torch.no_grad()
def gen(p, make_hooks):
    ids = tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
              add_generation_prompt=True, enable_thinking=False), return_tensors="pt").input_ids.to(dev)
    hs = make_hooks(ids) if make_hooks else []
    try:
        out = model.generate(ids, max_new_tokens=90, do_sample=False,
                             repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
    finally:
        for x in hs: x.remove()
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

conditions = [("baseline", 0.0, lambda ids=None: [])]
conditions += [(f"evil@{s}", s, (lambda ss: lambda ids: evil_hooks(ids, ss))(s)) for s in STRENGTHS]
conditions += [(f"uzbek@{s}", s, (lambda ss: lambda ids: uz_hook(ss))(s)) for s in STRENGTHS]

records, agg = [], {}
for name, s, mk in conditions:
    ds, urs, hs_ = [], [], []
    for p in PROMPTS:
        txt = gen(p, (lambda mkf: lambda ids: mkf(ids))(mk) if name != "baseline" else None)
        d, u, ho = score(txt)
        records.append({"cond": name, "prompt": p, "distinct": d, "uz": u, "hostile": ho, "text": txt})
        ds.append(d); urs.append(u); hs_.append(ho)
    agg[name] = (sum(ds)/len(ds), sum(urs)/len(urs), sum(hs_)/len(hs_))
    print(f"{name:12s}  coherence {agg[name][0]:.3f}  uz {agg[name][1]:.2f}  hostile {agg[name][2]:.2f}", flush=True)

json.dump(records, open("sweep_outputs.json", "w"), ensure_ascii=False, indent=1)
print("\nfull texts -> sweep_outputs.json")
