# 🧠 RLHF Text Optimization with PPO

This project implements a simplified **Reinforcement Learning from Human Feedback (RLHF)** pipeline using **Proximal Policy Optimization (PPO)** to improve text generation quality based on reward signals.

The system uses a pretrained language model as a policy and a sentiment classifier as a reward model to optimize generated outputs.

---

## 🚀 Features

- ✅ RLHF training loop using PPO
- ✅ Hugging Face TRL integration
- ✅ Reward modeling via sentiment analysis
- ✅ Before/After generation comparison
- ✅ Fully runnable in Google Colab

---

## 🧩 Project Structure


.
├── RLHF_HW(4).ipynb # Main training notebook
├── README.md # Project documentation
└── requirements.txt # Dependencies (optional)


---

## ⚙️ Methodology

This project follows a simplified RLHF pipeline:

### 1. Policy Model
- A pretrained language model (e.g. GPT-2)
- Generates text based on prompts

### 2. Reward Model
- A sentiment classifier (e.g. DistilBERT)
- Assigns scores:
  - Positive → higher reward
  - Negative → lower reward

### 3. PPO Optimization
- Updates the policy model to maximize reward
- Maintains stability via KL divergence penalty

---

## 🔁 Training Workflow

1. Input prompt → generate text
2. Evaluate text using reward model
3. Compute reward score
4. Update model via PPO
5. Repeat

---

## 📊 Example Output

### Before RLHF:
```
"The movie was okay but I didn't like the ending..."
```

### After RLHF:
```
"The movie was absolutely amazing and I loved every moment of it!"
```

---

## 🧪 Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face TRL (PPOTrainer)
- Google Colab

---

## 📦 Installation

```bash
pip install transformers datasets trl accelerate
▶️ Run the Project

You can run the notebook directly:

jupyter notebook RLHF_HW(4).ipynb

Or upload to Google Colab for easier execution.

⚠️ Known Issues
Dependency conflicts (e.g. numpy version)
PPO training may be slow without GPU
Reward model is simplified (sentiment ≠ human preference)
💡 Future Improvements
Replace sentiment reward with human-labeled data
Add multi-objective rewards (toxicity, helpfulness)
Improve model architecture (larger LLMs)
Add evaluation metrics (BLEU, ROUGE, human eval)
📚 Learning Goals

This project demonstrates:

How RLHF works in practice
How PPO is applied to language models
The role of reward models in LLM alignment
🧑‍💻 Author
Name: Fengyuan Liu
Program: Computer Science / AI
University: Purdue University Northwest
📜 License
MIT License
