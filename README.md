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
├── RLHF_HW_fixed.ipynb  # Main Colab/Jupyter notebook
├── README.md           # Project documentation
└── requirements.txt    # Pinned notebook dependencies
```

## Setup

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

For Google Colab, open the notebook and run the first dependency cell. After
that cell finishes, restart the runtime as instructed in the notebook, then run
the remaining cells from top to bottom.

## Run

```bash
jupyter notebook RLHF_HW_fixed.ipynb
```

The workflow:

1. Load the policy model and frozen reference model.
2. Build positive-review prompts from IMDB examples.
3. Generate policy responses.
4. Score only the generated response text with the reward model.
5. Update the policy with PPO.
6. Compare baseline and trained response rewards on held-out prompts.

## Known Limitations

- The reward model measures positive sentiment, not overall review quality.
- PPO training is slow without a GPU.
- Small models can over-optimize sentiment words instead of producing better
  reviews.
- Human preference labels and multi-objective rewards would make this closer to
  production RLHF.

## Author

Fengyuan Liu
