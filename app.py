# ==========================================
# 🤖 Empathetic Chatbot (Transformer-from-Scratch)
# ==========================================
import streamlit as st
import torch
import torch.nn.functional as F
import sentencepiece as spm
import math, json, os
from torch import nn

# ==== Load config.json to match Kaggle training exactly ====
if os.path.exists("config.json"):
    with open("config.json") as f:
        cfg = json.load(f)
    PAD_ID, BOS_ID, EOS_ID = cfg["PAD_ID"], cfg["BOS_ID"], cfg["EOS_ID"]
    VOCAB_SIZE = cfg["VOCAB_SIZE"]
    MAX_LEN_IN, MAX_LEN_OUT = cfg["MAX_LEN_IN"], cfg["MAX_LEN_OUT"]
    D_MODEL, HEADS, N_ENC, N_DEC, D_FF = cfg["D_MODEL"], cfg["HEADS"], cfg["N_ENC"], cfg["N_DEC"], cfg["D_FF"]
else:
    # Fallback in case config.json missing
    PAD_ID, BOS_ID, EOS_ID = 5, 3, 4
    VOCAB_SIZE = 16000
    MAX_LEN_IN, MAX_LEN_OUT = 128, 64
    D_MODEL, HEADS, N_ENC, N_DEC, D_FF = 256, 4, 3, 3, 1024

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== Transformer Architecture (must be identical to Kaggle training) ====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=D_MODEL, heads=HEADS, dropout=0.1):
        super().__init__()
        self.h, self.d_k = heads, d_model // heads
        self.q, self.k, self.v = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None):
        B, Tq, _ = q.shape
        h, dk = self.h, self.d_k
        q = self.q(q).view(B, Tq, h, dk).transpose(1, 2)
        k = self.k(k).view(B, -1, h, dk).transpose(1, 2)
        v = self.v(v).view(B, -1, h, dk).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
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

# ==== Load tokenizer & model ====
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
        if len(seen) >= 2 and w == seen[-1] == seen[-2]:
            continue
        seen.append(w)
    return " ".join(seen)

# ==== Greedy Decoding ====
@torch.no_grad()
def greedy_text(model, x_text, max_len=64):
    model.eval()
    src = torch.tensor([encode_str(x_text, add_bos=True, add_eos=True, max_len=MAX_LEN_IN)], device=DEVICE)
    ys = torch.tensor([[BOS_ID]], device=DEVICE)
    for _ in range(max_len):
        logits = model(src, ys)
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        ys = torch.cat([ys, nxt], dim=1)
        if nxt.item() == EOS_ID: break
    return clean_text(ids_to_text(ys.squeeze(0).tolist()))

# ==== Beam Search Decoding ====
@torch.no_grad()
def beam_text(model, x_text, beam=4, max_len=64, lp_alpha=0.7):
    model.eval()
    src = torch.tensor([encode_str(x_text, add_bos=True, add_eos=True, max_len=MAX_LEN_IN)], device=DEVICE)
    beams = [(torch.tensor([[BOS_ID]], device=DEVICE), 0.0)]
    for _ in range(max_len):
        new_beams = []
        for ys, score in beams:
            logits = model(src, ys)
            logp = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
            topk = torch.topk(logp, beam)
            for tok, lp in zip(topk.indices, topk.values):
                ys2 = torch.cat([ys, tok.view(1, 1)], dim=1)
                new_beams.append((ys2, score + lp.item()))
        def lp_fn(sc, L): return sc / (((5 + L) ** lp_alpha) / ((5 + 1) ** lp_alpha))
        beams = sorted(new_beams, key=lambda t: lp_fn(t[1], t[0].size(1)), reverse=True)[:beam]
        if any(b[0][0, -1].item() == EOS_ID for b in beams): break
    best = max(beams, key=lambda t: t[1])[0]
    return clean_text(ids_to_text(best.squeeze(0).tolist()))

# ==== Streamlit UI ====
st.title("🤖 Empathetic Chatbot (Transformer-from-Scratch)")
st.caption("Built entirely from scratch on the Empathetic Dialogues dataset (same as Kaggle).")

emotion = st.text_input("Emotion", "grateful")
situation = st.text_input("Situation", "i passed my exam today")
customer = st.text_area("Customer Utterance", "i am so happy! i studied so hard and finally passed")
strategy = st.radio("Decoding Strategy", ["Greedy", "Beam"], horizontal=True)

if st.button("Submit"):
    prompt = f"emotion: {emotion} | situation: {situation} | customer: {customer} agent:"
    if strategy == "Greedy":
        reply = greedy_text(model, prompt)
    else:
        reply = beam_text(model, prompt)
    st.success(reply)

st.caption("Built with ❤️ using Streamlit")
