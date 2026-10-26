"""Interactive terminal chat with Qwen3.5-4B on this Mac (MPS), streaming replies.

Usage: .venv/bin/python chat.py [--think] [--steer] [--uzbek] [--uzbek-neuron]
Commands: /reset clears history, /steer toggles the original pair, /uzbek toggles language-vector
steering, /uzbek-neuron toggles L31 N379, /invert reverses active steering directions, /exit quits.
"""
import argparse, sys, threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL = "Qwen/Qwen3.5-4B"
STEERS = [(14, 8949, +1.0), (13, 7956, -1.0)]
# Uzbek-selective detector neuron, available as an independent experimental steer.
UZBEK_NEURON = (31, 379, +1.0)
UZBEK_SENTENCES = [
    "Men bugun bozorga bordim va non sotib oldim.",
    "Kitob oʻqish eng sevimli mashgʻulotim.",
    "Ertaga maktabga erta borishim kerak.",
    "Bugun havo juda issiq, quyosh charaqlab turibdi.",
    "Toshkent Oʻzbekistonning poytaxti hisoblanadi.",
    "Oilam bilan choy ichib, suhbatlashdik.",
]
ENGLISH_SENTENCES = [
    "I went to the market today and bought bread.",
    "Reading books is my favorite activity.",
    "Tomorrow I need to go to school early.",
    "The weather is very hot today, the sun is shining.",
    "Tashkent is the capital of Uzbekistan.",
    "We drank tea and chatted with my family.",
]

ap = argparse.ArgumentParser()
ap.add_argument("--think", action="store_true", help="let the model think before answering (slower)")
ap.add_argument("--steer", action="store_true", help="enable neuron steering")
ap.add_argument("--uzbek", action="store_true", help="enable Uzbek language-vector steering")
ap.add_argument("--uzbek-coeff", type=float, default=1.0, help="Uzbek vector strength")
ap.add_argument("--uzbek-layer", type=int, default=16, help="layer for Uzbek vector")
ap.add_argument("--uzbek-neuron", action="store_true", help="enable Uzbek-selective neuron L31 N379")
ap.add_argument("--max-new-tokens", type=int, default=1024)
ap.add_argument("--temperature", type=float, default=0.7)
args = ap.parse_args()

dev = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"loading {MODEL} on {dev} ...", file=sys.stderr, flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()

# Locate the decoder layers inside Qwen's model wrapper.
layers = next((mod for mod in model.modules()
               if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 24
               and hasattr(mod[0], "mlp")), None)
if layers is None:
    raise RuntimeError("could not locate decoder layers for steering")

@torch.no_grad()
def median_resid_norm(input_ids, steers):
    """Median L2 norm of each steered layer's output over positions 1:."""
    caught = {}
    def make_hook(layer_index):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            caught[layer_index] = hidden.detach()
        return hook

    target_layers = {layer for layer, _, _ in steers}
    handles = [layers[layer].register_forward_hook(make_hook(layer))
               for layer in target_layers]
    try:
        model(input_ids)
    finally:
        for handle in handles:
            handle.remove()

    norms = {}
    for layer, hidden in caught.items():
        norms[layer] = hidden[0, 1:].float().norm(dim=-1).median().item()
    return norms

def make_steering_hook(offset, neuron):
    def hook(_module, inputs):
        hidden = inputs[0].clone()
        hidden[..., neuron] += offset
        return (hidden,) + inputs[1:]
    return hook

@torch.no_grad()
def build_uzbek_vector(layer_index):
    """Build an Uzbek-minus-English residual direction from matched sentences."""
    captured = {}
    def capture(_module, _inputs, output):
        captured[0] = output[0] if isinstance(output, tuple) else output

    handle = layers[layer_index].register_forward_hook(capture)
    hidden_size = layers[0].mlp.down_proj.weight.shape[0]
    def mean_activation(sentences):
        total = torch.zeros(hidden_size, dtype=torch.float32)
        for sentence in sentences:
            sample_ids = tok(sentence, return_tensors="pt").input_ids.to(dev)
            model(sample_ids)
            total += captured[0][0, 1:].float().mean(0).cpu()
        return total / len(sentences)

    try:
        vector = mean_activation(UZBEK_SENTENCES) - mean_activation(ENGLISH_SENTENCES)
    finally:
        handle.remove()
    return vector.to(dev)

print("ready. /reset clears history, /steer toggles the original neuron pair, /uzbek toggles "
      "the Uzbek vector, /uzbek-neuron toggles L31 N379, /invert reverses active directions, "
      "/exit quits.\n",
      file=sys.stderr, flush=True)

history = []
steering_enabled = args.steer
uzbek_vector_enabled = args.uzbek
uzbek_neuron_enabled = args.uzbek_neuron
steering_inverted = False
uzbek_vector = None
while True:
    try:
        user = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not user:
        continue
    if user == "/exit":
        break
    if user == "/reset":
        history.clear()
        print("(history cleared)\n")
        continue
    if user == "/steer":
        steering_enabled = not steering_enabled
        print(f"(steering {'on' if steering_enabled else 'off'})\n")
        continue
    if user == "/invert":
        steering_inverted = not steering_inverted
        print(f"(steering direction {'inverted' if steering_inverted else 'normal'})\n")
        continue
    if user == "/uzbek":
        uzbek_vector_enabled = not uzbek_vector_enabled
        print(f"(Uzbek vector {'on' if uzbek_vector_enabled else 'off'})\n")
        continue
    if user == "/uzbek-neuron":
        uzbek_neuron_enabled = not uzbek_neuron_enabled
        print(f"(Uzbek neuron L{UZBEK_NEURON[0]} N{UZBEK_NEURON[1]} "
              f"{'on' if uzbek_neuron_enabled else 'off'})\n")
        continue
    if user.startswith("/uzbek "):
        try:
            args.uzbek_coeff = float(user.split(maxsplit=1)[1])
            uzbek_vector_enabled = args.uzbek_coeff != 0
            print(f"(Uzbek vector coeff={args.uzbek_coeff:g})\n")
        except ValueError:
            print("usage: /uzbek <coefficient>\n")
        continue

    history.append({"role": "user", "content": user})
    prompt = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=args.think)
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen = dict(input_ids=ids, streamer=streamer, max_new_tokens=args.max_new_tokens,
               do_sample=args.temperature > 0, temperature=args.temperature or None, top_p=0.9,
               repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
    handles = []
    active_steers = []
    if steering_enabled:
        active_steers.extend(STEERS)
    if uzbek_neuron_enabled:
        active_steers.append(UZBEK_NEURON)

    if uzbek_vector_enabled and args.uzbek_coeff != 0:
        if not 0 <= args.uzbek_layer < len(layers):
            raise ValueError(f"--uzbek-layer must be between 0 and {len(layers) - 1}")
        if uzbek_vector is None:
            print("building Uzbek steering vector...", file=sys.stderr, flush=True)
            uzbek_vector = build_uzbek_vector(args.uzbek_layer)
            print(f"Uzbek vector ready (norm {uzbek_vector.norm():.2f})",
                  file=sys.stderr, flush=True)

    if active_steers:
        if any(layer >= len(layers) for layer, _, _ in active_steers):
            raise RuntimeError(f"model has {len(layers)} layers; an active steer references a missing layer")
        norms = median_resid_norm(ids, active_steers)
        for layer, neuron, dose in active_steers:
            if steering_inverted:
                dose = -dose
            column_norm = layers[layer].mlp.down_proj.weight[:, neuron].float().norm().item()
            offset = dose * norms[layer] / column_norm
            handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(
                make_steering_hook(offset, neuron)))
            print(f"  steer L{layer} N{neuron} dose{dose:+g}: offset={offset:+.3f}",
                  file=sys.stderr, flush=True)
    if uzbek_vector_enabled and args.uzbek_coeff != 0:
        coeff = -args.uzbek_coeff if steering_inverted else args.uzbek_coeff
        def add_uzbek_vector(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            steered = hidden + coeff * uzbek_vector.to(hidden.dtype)
            return (steered,) + output[1:] if isinstance(output, tuple) else steered
        handles.append(layers[args.uzbek_layer].register_forward_hook(add_uzbek_vector))
        print(f"  Uzbek vector L{args.uzbek_layer} coeff={coeff:+g}",
              file=sys.stderr, flush=True)

    def run_generation():
        try:
            with torch.no_grad():
                model.generate(**gen)
        finally:
            for handle in handles:
                handle.remove()

    threading.Thread(target=run_generation, daemon=True).start()

    print("qwen> ", end="", flush=True)
    reply = []
    try:
        for piece in streamer:
            print(piece, end="", flush=True)
            reply.append(piece)
    except KeyboardInterrupt:
        print(" [interrupted]", end="")
    print("\n")
    history.append({"role": "assistant", "content": "".join(reply).strip()})
