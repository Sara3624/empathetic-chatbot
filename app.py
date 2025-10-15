import streamlit as st
import torch
import torch.nn.functional as F
import sentencepiece as spm
from torch import nn
import math


# CONFIG
PAD_ID, BOS_ID, EOS_ID = 5, 3, 4
VOCAB_SIZE = 16000
MAX_LEN_IN, MAX_LEN_OUT = 64


# MODEL ARCHITECTURE
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, heads=4, dropout=0.1):
        super().__init__()
        self.h, self.d_k = heads, d_model // heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
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
            scores = scores.masked_fill(mask, float('-inf'))
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
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model=256, heads=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
    def forward(self, x, src_mask):
        x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), src_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, heads=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
    def forward(self, y, mem, tgt_mask, mem_mask):
        y = y + self.self_attn(self.ln1(y), self.ln1(y), self.ln1(y), tgt_mask)
        y = y + self.cross_attn(self.ln2(y), mem, mem, mem_mask)
        y = y + self.ffn(self.ln3(y))
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
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(n_enc)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(n_dec)])
        self.ln_enc = nn.LayerNorm(d_model)
        self.ln_dec = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src_mask, tgt_pad = make_pad_mask(src), make_pad_mask(tgt)
        causal = make_causal_mask(tgt.size(1))
        tgt_mask = tgt_pad | causal

        mem = self.pos(self.embed(src))
        for layer in self.enc_layers:
            mem = layer(mem, src_mask)
        mem = self.ln_enc(mem)

        y = self.pos(self.embed(tgt))
        for layer in self.dec_layers:
            y = layer(y, mem, tgt_mask, src_mask)
        y = self.ln_dec(y)
        return self.out(y)

# LOAD TOKENIZER & MODEL
sp = spm.SentencePieceProcessor(model_file="spm.model")
model = TransformerChat()
state = torch.load("model.pt", map_location="cpu")
model.load_state_dict(state, strict=False)
model.eval()

# TEXT HELPERS
def encode_str(s, add_bos=False, add_eos=False, max_len=128):
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

# DECODE METHODS
@torch.no_grad()
def greedy_decode(src):
    ys = torch.tensor([[BOS_ID]])
    for _ in range(MAX_LEN_OUT):
        logits = model(src, ys)
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        ys = torch.cat([ys, nxt], dim=1)
        if nxt.item() == EOS_ID:
            break
    return ys.squeeze(0).tolist()


@torch.no_grad()
def beam_decode(src, beam_width=3):
    beams = [(torch.tensor([[BOS_ID]]), 0.0)]  # (sequence, logprob)
    for _ in range(MAX_LEN_OUT):
        new_beams = []
        for ys, score in beams:
            logits = model(src, ys)
            probs = F.log_softmax(logits[:, -1, :], dim=-1)
            topk = torch.topk(probs, beam_width)
            for k in range(beam_width):
                next_tok = topk.indices[0][k].view(1, 1)
                new_seq = torch.cat([ys, next_tok], dim=1)
                new_score = score + topk.values[0][k].item()
                new_beams.append((new_seq, new_score))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        if all(seq[0, -1].item() == EOS_ID for seq, _ in beams):
            break
    best_seq = beams[0][0].squeeze(0).tolist()
    return best_seq


# GENERATE REPLY
def generate_reply(emotion, situation, customer, mode="Greedy"):
    x = f"emotion: {emotion} | situation: {situation} | customer: {customer} agent:"
    src = torch.tensor([encode_str(x, add_bos=True, add_eos=True, max_len=MAX_LEN_IN)])
    if mode == "Greedy":
        out_ids = greedy_decode(src)
    else:
        out_ids = beam_decode(src)
    return ids_to_text(out_ids)


# STREAMLIT UI
st.title("Empathetic Chatbot (Transformer-from-Scratch)")
st.write("Built entirely from scratch on the Empathetic Dialogues dataset!")

emotion = st.text_input("Emotion", "grateful")
situation = st.text_input("Situation", "I passed my exam today!")
customer = st.text_area("Customer Message", "I'm so happy! I studied hard and finally passed.")
mode = st.selectbox("Choose Decoding Strategy", ["Greedy", "Beam Search"])

if st.button("Generate Reply"):
    with st.spinner("Generating reply..."):
        reply = generate_reply(emotion, situation, customer, mode)
    st.success(f"**{mode} Output:** {reply}")
