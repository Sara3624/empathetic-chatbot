🧠 Empathetic Chatbot (Transformer-from-Scratch)
This project implements a Transformer-based encoder–decoder chatbot trained from scratch on the [Empathetic Dialogues Dataset](https://www.kaggle.com/datasets/atharvjairath/empathetic-dialogues-facebook-ai).
No pretrained weights are used — all model parameters are randomly initialized and trained end-to-end on Kaggle GPU.


🌟 Project Overview
The chatbot is designed to generate empathetic agent responses based on:
* Emotion of the user
* Situation describing the context
* Customer utterance

Example input → output:
> Input:
> Emotion: sad
> Situation: I failed my driving test again.
> Customer: I feel like giving up after trying so many times.
>
> Output:
> “Don’t be too hard on yourself, you’ll get it next time. Everyone struggles sometimes.”

⚙️ Model Architecture
The chatbot uses a custom Transformer Encoder–Decoder implemented entirely from scratch using PyTorch:
* Multi-Head Attention
* Positional Encoding
* Layer Normalization
* Feed-Forward Networks
* Teacher-Forcing training
* Greedy & Beam Search decoding

🧩 Dataset
Dataset: `emotion-emotion_69k.csv` (Empathetic Dialogues)
Cleaned and preprocessed into four key columns:
* `emotion`
* `situation`
* `customer` (extracted from dialogue)
* `agent` (target reply)

🧠 Training Details
| Parameter  | Value                           |
| ---------- | ------------------------------- |
| Epochs     | 5–15                            |
| Batch Size | 32                              |
| Optimizer  | Adam (lr=3e-4)                  |
| Loss       | CrossEntropyLoss                |
| Tokenizer  | SentencePiece (BPE, vocab=16k)  |
| Metrics    | BLEU, ROUGE-L, chrF, Perplexity |

Example results after 5 epochs:

| Metric     | Validation | Test |
| ---------- | ---------- | ---- |
| BLEU       | 1.84       | 1.83 |
| ROUGE-L    | 15.0       | 15.0 |
| chrF       | 11.8       | 11.7 |
| Perplexity | 43.7       | 43.9 |

💬 Interactive Demo
Try the chatbot live on Streamlit:
👉[Empathetic Chatbot Demo](https://empathetic-chatbot.streamlit.app)
Input your emotion, situation, and message — and get an empathetic response from the trained Transformer model.

### 🚀 How to Run Locally

```bash
# Clone repo
git clone https://github.com/Sara3624/empathetic-chatbot.git
cd empathetic-chatbot

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

📦 Project Structure
```
empathetic-chatbot/
├── app.py                # Streamlit UI
├── model.pt              # Trained Transformer weights
├── spm.model             # SentencePiece tokenizer
├── requirements.txt       # Dependencies
└── README.md             # Documentation
```

📚 References
* Rashkin et al., Empathetic Dialogues: A Dataset for Training Empathetic Conversational Agents, ACL 2019
* Vaswani et al., Attention Is All You Need, NeurIPS 2017
* SentencePiece Tokenizer Documentation

-------------------------------------------------------------------------------------------------------------------------------------
👩‍💻 Author
Sara (FAST-NUCES, Batch 22)
Course: Generative AI – Fall 2025
Instructor: Dr. mohammad Usama
Platform: Kaggle + Streamlit Cloud Deployment
