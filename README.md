# qwen-steer

Neuron / activation steering experiments on Qwen/Qwen3.5-4B (runs on Mac MPS or CPU).

Setup:
    uv venv --python 3.12 .venv
    uv pip install -r requirements.txt
    # downloads ~6 GB of weights on first run

Scripts (run with `.venv/bin/python <script>`):
  steer.py       Demo: steer two MLP neurons and show the output shift.
  invert.py      Same two neurons with signs flipped (direction is not symmetric).
  uzbek.py       Find the neuron most selective for Uzbek (contrastive activation search).
  uzbek_chat.py  Interactive chat steered to answer in Uzbek via a direction vector (/coeff to tune).
  chat.py        Plain interactive chat with the base model.
  bake_uzbek.py  Bake the Uzbek direction into the weights as an o_proj bias (persists through reload).

Finding: a single neuron can *detect* a language but not *drive* it; forcing output
language needs the mean-difference direction (Uzbek - English) added mid-network.
