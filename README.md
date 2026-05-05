# RLHF Text Optimization with PPO

This project implements a compact Reinforcement Learning from Human Feedback
(RLHF) workflow that nudges GPT-2 style movie-review generations toward a
positive tone using Proximal Policy Optimization (PPO).

The notebook uses:

- `distilgpt2` as the policy model.
- `lvwerra/distilbert-imdb` as a sentiment-based reward model.
- Hugging Face TRL's `PPOTrainer` for policy updates.

> Sentiment is a simplified preference proxy, not a replacement for
> human-labeled preference data.

## Project Structure

```text
.
├── RLHF_HW_fixed.ipynb  # Main notebook (optimized for Kaggle)
├── README.md           # Project documentation
└── requirements.txt    # Pinned dependencies for local environments
```

## Setup

Python 3.10+ is recommended.

### Local environment

```bash
pip install -r requirements.txt
```

Dependencies are pinned for reproducibility. **`datasets==2.19.0`** is used (instead of 2.20.x)
because newer `datasets` builds can trigger `np.array(..., copy=False)` failures when paired with
NumPy 1.26.x during PyTorch-format dataset iteration (see Troubleshooting).

### Kaggle

1. Go to [kaggle.com/code](https://www.kaggle.com/code).
2. Create a new notebook.
3. Upload `RLHF_HW_fixed.ipynb`.
4. Turn on GPU: **Settings → Accelerator → GPU T4 x2**.
5. Click **Run All**.
6. **No restart needed** — Kaggle pre-installs `trl`, `transformers`, `accelerate`, and `datasets`.

## Run

### Kaggle/Colab

Open the uploaded notebook, enable **GPU T4 x2**, then **Run All** from the first cell.

### Local Jupyter

```bash
jupyter notebook RLHF_HW_fixed.ipynb
```

Use a GPU when available; PPO training is very slow on CPU.

The workflow:

1. Load the policy model and frozen reference model.
2. Build positive-review prompts from IMDB examples.
3. Generate policy responses.
4. Score only the generated response text with the reward model.
5. Update the policy with PPO.
6. Compare baseline and trained response rewards on held-out prompts.
7. **BERT before/after analysis**: run the DistilBERT sentiment classifier
   (`lvwerra/distilbert-imdb`) on at least 15 (default 20) baseline vs
   trained generations and produce a side-by-side comparison table with
   explicit `POSITIVE` / `NEGATIVE` labels, classifier confidences,
   per-prompt reward deltas, and a `NEG→POS` / `POS→NEG` flip column,
   plus aggregate flip-rate and mean-reward statistics.
(There is an output in Fixed file)

## Known Limitations

- The reward model measures positive sentiment, not overall review quality.
- PPO training is slow without a GPU.
- Small models can over-optimize sentiment words instead of producing better
  reviews.
- Human preference labels and multi-objective rewards would make this closer to
  production RLHF.

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'trl'` | Missing packages locally | Run `pip install -r requirements.txt` |
| `ValueError: Unable to avoid copy while creating an array as requested` | `datasets` / NumPy incompatibility when iterating `PPOTrainer`’s dataloader | Local: ensure **`datasets==2.19.0`** via `requirements.txt`; on Kaggle, add a one-off `pip install 'datasets==2.19.0'` cell if the default image regresses |
| `CUDA out of memory` | Batch size too large | Reduce `batch_size` in `PPOConfig` to `4` |
| Training is very slow | No GPU selected | Kaggle: **Settings → Accelerator → GPU T4 x2** |

## Author

Fengyuan Liu
