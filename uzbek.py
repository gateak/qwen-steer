"""Find the neuron(s) most selective for Uzbek.

Method: content-matched sentences (same meaning) in Uzbek vs a contrast set
(English, Russian, Turkish). For every MLP neuron (down_proj input, the same
"neuron" space the steering uses), average its activation over tokens 1: and
over sentences, per class. Rank neurons by (mean_uzbek - mean_contrast):
a high score = fires on Uzbek but not on the same sentence in other languages.
Turkish is in the contrast so we don't just find a generic Turkic neuron.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-4B"

UZBEK = [
    "Men bugun bozorga bordim va non sotib oldim.",
    "Kitob oʻqish eng sevimli mashgʻulotim.",
    "Ertaga maktabga erta borishim kerak.",
    "Bugun havo juda issiq, quyosh charaqlab turibdi.",
    "Toshkent Oʻzbekistonning poytaxti hisoblanadi.",
    "Oilam bilan choy ichib, suhbatlashdik.",
]
CONTRAST = [
    # English
    "I went to the market today and bought bread.",
    "Reading books is my favorite activity.",
    "Tomorrow I need to go to school early.",
    "The weather is very hot today, the sun is shining.",
    # Russian
    "Сегодня я ходил на рынок и купил хлеб.",
    "Чтение книг — моё любимое занятие.",
    "Завтра мне нужно рано пойти в школу.",
    "Сегодня очень жарко, светит солнце.",
    # Turkish (Turkic neighbour: keeps the result Uzbek-specific)
    "Bugün pazara gittim ve ekmek aldım.",
    "Kitap okumak en sevdiğim aktivite.",
    "Yarın okula erken gitmem gerekiyor.",
    "Bugün hava çok sıcak, güneş parlıyor.",
]

dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
layers = next(m for m in model.modules()
              if isinstance(m, torch.nn.ModuleList) and len(m) >= 24 and hasattr(m[0], "mlp"))
n_layers, inter = len(layers), layers[0].mlp.down_proj.weight.shape[1]
print(f"{n_layers} layers x {inter} neurons", flush=True)

latest = {}
def pre(L):
    return lambda m, i: latest.__setitem__(L, i[0].detach())
handles = [layers[L].mlp.down_proj.register_forward_pre_hook(pre(L)) for L in range(n_layers)]

@torch.no_grad()
def class_means(sentences):
    acc = torch.zeros(n_layers, inter, dtype=torch.float32)
    for s in sentences:
        ids = tok(s, return_tensors="pt").input_ids.to(dev)
        model(ids)
        for L in range(n_layers):
            acc[L] += latest[L][0, 1:].float().mean(dim=0).cpu()   # mean over tokens 1:
    return acc / len(sentences)

uz = class_means(UZBEK)
ct = class_means(CONTRAST)
for h in handles: h.remove()

diff = uz - ct
flat = diff.flatten()
topk = torch.topk(flat, 15)
print("\nrank  layer  neuron   uzbek_mean  contrast_mean   diff")
for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
    L, n = divmod(idx, inter)
    print(f"      L{L:<4d} N{n:<6d} {uz[L, n]:+9.3f}   {ct[L, n]:+9.3f}   {score:+8.3f}")

# Also the single most selective, and its neighbours in the same layer for context.
L, n = divmod(int(flat.argmax()), inter)
print(f"\nTop Uzbek neuron: L{L} N{n}  (fires {uz[L,n]:+.2f} on Uzbek vs {ct[L,n]:+.2f} on the same sentences elsewhere)")
