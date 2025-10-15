# ==========================================
# 🤖 Empathetic Chatbot (Transformer-from-Scratch)
# ==========================================
import streamlit as st
import torch
import torch.nn.functional as F
import sentencepiece as spm
import math, json, os
from torch import nn

# ==== Load config.json to ensure matching hyperparameters ====
if os.path.exists("config.json"):
    with open("config.json") as f:
        cfg = json.load(f)
    PAD_ID, BOS_ID, EOS_ID = cfg["PAD_ID"], cfg["BOS_ID"], cfg["EOS_ID"]
    VOCAB_SIZE = cfg["VOCAB_SIZE"]
    MAX_LEN_IN, MAX_LEN_OUT = cfg["MAX_LEN_IN"], cfg["MAX_LEN_OUT"]
    D_MODEL, HEADS, N_ENC, N_DEC, D_FF = cfg["D_MODEL"], cfg["HEADS"], cfg["N_ENC"], cfg["N_DEC"], cfg["D_FF"]
else:
    # fallback (in case config.json missing)
    PAD_ID, BOS_ID, EOS_ID = 5, 3, 4
    VOCAB_SIZE = 16000
    MAX_LEN_IN, MAX_LEN_OUT = 128, 64
    D_MODEL, HEADS, N_ENC, N_DEC, D_FF = 256, 4, 3, 3, 1024

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== Model Architecture (identical to training) ====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=D_MODEL, heads=HEADS, dropout=0.1):
        super().__init__()
        self.h, self.d_k = heads, d_model // heads
        self.q, self.k, self.v = [nn.Linear(d_model, d_model) for _ in range(3)]
        self.q, self.k, self.v = nn.Linear(d_model,d_model), nn.Linear(d_model,d_model), nn.Linear(d_model,d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None):
        B, Tq, _ = q.shape
        h, dk = self.h, self.d_k
        q = self.q(q).view(B, Tq, h, dk).transpose(1, 2)
        k = self.k(k).view(B, -1, h, dk).transpose(1, 2)
        v = self.v(v).view(B, -1, h, dk).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None: scores = scores.masked_fill(mask, float("-inf"))
        attn = self.drop(F.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, Tq, h * dk)
        return self.o(out)

class FFN(nn.Module):
    def __init__(self, d_model=D_MODEL, d_ff=D_FF, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ff = FFN()
        self.l1, self.l2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
    def forward(self, x, mask):
        x = x + self.sa(self.l1(x), self.l1(x), self.l1(x), mask)
        x = x + self.ff(self.l2(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ca = MultiHeadAttention()
        self.ff = FFN()
        self.l1, self.l2, self.l3 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
    def forward(self, y, mem, tgt_mask, mem_mask):
        y = y + self.sa(self.l1(y), self.l1(y), self.l1(y), tgt_mask)
        y = y + self.ca(self.l2(y), mem, mem, mem_mask)
        y = y + self.ff(self.l3(y))
        return y

def make_pad_mask(seq, pad=PAD_ID): return (seq == pad).unsqueeze(1).unsqueeze(2)
def make_causal_mask(sz): return torch.triu(torch.ones(sz, sz, dtype=torch.bool), 1).unsqueeze(0).unsqueeze(0)

class TransformerChat(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(D_MODEL)
        self.encs = nn.ModuleList([EncoderLayer() for _ in range(N_ENC)])
        self.decs = nn.ModuleList([DecoderLayer() for _ in range(N_DEC)])
        self.lnE, self.lnD = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.out = nn.Linear(D_MODEL, VOCAB_SIZE)
    def forward(self, src, tgt):
        spad, tpad = make_pad_mask(src), make_pad_mask(tgt)
        causal = make_causal_mask(tgt.size(1))
        tmask = tpad | causal
        mem = self.pos(self.embed(src))
        for l in self.encs: mem = l(mem, spad)
        y = self.pos(self.embed(tgt))
        for l in self.decs: y = l(y, mem, tmask, spad)
        return self.out(self.lnD(y))

# ==== Load Tokenizer & Model ====
sp = spm.SentencePieceProcessor(model_file="spm.model")
model = TransformerChat().to(DEVICE)
state = torch.load("model.pt", map_location=DEVICE)
model.load_state_dict(state, strict=False)
model.eval()

# ==== Helpers ====
def encode_str(s, add_bos=False, add_eos=False, max_len=MAX_LEN_IN):
    ids = sp.encode(s, out_type=int)
    if add_bos: ids = [BOS_ID] + ids
    if add_eos: ids = ids + [EOS_ID]
    return ids[:max_len]

def ids_to_text(ids):
    toks = []
    for t in ids:
        if t in (PAD_ID, BOS_ID): continue
        if t == EOS_ID: break
        toks.append(int(t))
    return sp.decode(toks)

def clean_text(text):
    words = text.split()
    seen = []
    for w in words:
        if len(seen) >= 2 and w == seen[-1] == seen[-2]: continue
        seen.append(w)
    text = " ".join(seen)
    return text[:500]

# ==== Nucleus Sampling Decoding ====
@torch.no_grad()
def sample_text(model, x_text, top_p=0.9, temperature=1.0, max_len=MAX_LEN_OUT):
    model.eval()
    src = torch.tensor([encode_str(x_text, add_bos=True, add_eos=True)], device=DEVICE)
    ys = torch.tensor([[BOS_ID]], device=DEVICE)
    for _ in range(max_len):
        logits = model(src, ys)[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative_probs > top_p
        sorted_probs[cutoff] = 0
        sorted_probs /= sorted_probs.sum()
        next_token = torch.multinomial(sorted_probs, 1)
        next_token = sorted_idx.gather(-1, next_token)
        ys = torch.cat([ys, next_token], dim=1)
        if next_token.item() == EOS_ID: break
    return clean_text(ids_to_text(ys.squeeze(0).tolist()))

# ==== Streamlit UI ====
st.title("🤖 Empathetic Chatbot (Transformer-from-Scratch)")
st.caption("Built and trained completely from scratch on the Empathetic Dialogues dataset.")

emotion = st.text_input("Emotion", "grateful")
situation = st.text_input("Situation", "I passed my exam today!")
customer = st.text_area("Customer Message", "I'm so happy! I studied hard and finally passed.")
strategy = st.radio("Decoding Strategy", ["Nucleus Sampling", "Beam Search"], horizontal=True)

if st.button("Generate Reply"):
    prompt = f"emotion: {emotion} | situation: {situation} | customer: {customer} agent:"
    if strategy == "Beam Search":
        reply = sample_text(model, prompt, top_p=1.0, temperature=1.2)
    else:
        reply = sample_text(model, prompt, top_p=0.9, temperature=1.0)
    st.success(reply)
