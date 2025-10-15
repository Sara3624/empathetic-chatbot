import streamlit as st
import torch
import torch.nn.functional as F
import sentencepiece as spm
import math
from torch import nn

# ==== Constants ====
PAD_ID, BOS_ID, EOS_ID = 5, 3, 4
VOCAB_SIZE = 16000
MAX_LEN_IN, MAX_LEN_OUT = 128, 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==== Model Architecture (same as training) ====
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
    def __init__(self, d_model=256, heads=4, dropout=0.1):
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
    def __init__(self, d_model=256, d_ff=1024, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model=256, heads=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.sa = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FFN(d_model, d_ff, dropout)
        self.l1, self.l2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + self.sa(self.l1(x), self.l1(x), self.l1(x), mask)
        x = x + self.ff(self.l2(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, heads=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.sa = MultiHeadAttention(d_model, heads, dropout)
        self.ca = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FFN(d_model, d_ff, dropout)
        self.l1, self.l2, self.l3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, y, mem, tgt_mask, mem_mask):
        y = y + self.sa(self.l1(y), self.l1(y), self.l1(y), tgt_mask)
        y = y + self.ca(self.l2(y), mem, mem, mem_mask)
        y = y + self.ff(self.l3(y))
        return y

def make_pad_mask(seq, pad=PAD_ID):
    return (seq == pad).unsqueeze(1).unsqueeze(2)

def make_causal_mask(sz):
    return torch.triu(torch.ones(sz, sz, dtype=torch.bool), 1).unsqueeze(0).unsqueeze(0)

class TransformerChat(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=256, heads=4, n_enc=3, n_dec=3, d_ff=1024, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model)
        self.encs = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(n_enc)])
        self.decs = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(n_dec)])
        self.lnE, self.lnD = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        spad, tpad = make_pad_mask(src), make_pad_mask(tgt)
        causal = make_causal_mask(tgt.size(1))
        tmask = tpad | causal
        mem = self.pos(self.embed(src))
        for l in self.encs:
            mem = l(mem, spad)
        y = self.pos(self.embed(tgt))
        for l in self.decs:
            y = l(y, mem, tmask, spad)
        return self.out(self.lnD(y))

# ==== Load tokenizer & model ====
sp = spm.SentencePieceProcessor(model_file="spm.model")
model = TransformerChat().to(DEVICE)
state = torch.load("model.pt", map_location=DEVICE)
model.load_state_dict(state, strict=False)
model.eval()

# ==== Encoding / Decoding Helpers ====
def encode_str(s, add_bos=False, add_eos=False, max_len=128):
    ids = sp.encode(s, out_type=int)
    if add_bos:
        ids = [BOS_ID] + ids
    if add_eos:
        ids = ids + [EOS_ID]
    return ids[:max_len]

def ids_to_text(ids):
    toks = []
    for t in ids:
        if t in (PAD_ID, BOS_ID):
            continue
        if t == EOS_ID:
            break
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

# ==== Beam Search ====
@torch.no_grad()
def beam_text(model, x_text, beam=4, max_len=64, lp_alpha=0.7, temperature=1.2):
    model.eval()
    src = torch.tensor([encode_str(x_text, add_bos=True, add_eos=True, max_len=MAX_LEN_IN)], device=DEVICE)
    beams = [(torch.tensor([[BOS_ID]], device=DEVICE), 0.0)]
    for _ in range(max_len):
        new_beams = []
        for ys, score in beams:
            logits = model(src, ys)
            logp = F.log_softmax(logits[:, -1, :] / temperature, dim=-1).squeeze(0)
            topk = torch.topk(logp, beam)
            for tok, lp in zip(topk.indices, topk.values):
                ys2 = torch.cat([ys, tok.view(1, 1)], dim=1)
                new_beams.append((ys2, score + lp.item()))
        def lp_fn(sc, L):
            return sc / (((5 + L) ** lp_alpha) / ((5 + 1) ** lp_alpha))
        beams = sorted(new_beams, key=lambda t: lp_fn(t[1], t[0].size(1)), reverse=True)[:beam]
        if any(b[0][0, -1].item() == EOS_ID for b in beams):
            break
    best = max(beams, key=lambda t: t[1])[0]
    return clean_text(ids_to_text(best.squeeze(0).tolist()))

# ==== Streamlit UI ====
st.title("🤖 Empathetic Chatbot (Transformer-from-Scratch)")
st.write("Built entirely from scratch on the Empathetic Dialogues dataset!")

emotion = st.text_input("Emotion", "grateful")
situation = st.text_input("Situation", "I passed my exam today!")
customer = st.text_area("Customer Message", "I'm so happy! I studied hard and finally passed.")

if st.button("Generate Reply"):
    input_text = f"emotion: {emotion} | situation: {situation} | customer: {customer} agent:"
    reply = beam_text(model, input_text)
    st.success(reply)
