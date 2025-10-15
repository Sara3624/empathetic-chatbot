import os
import math
import streamlit as st
import torch
import torch.nn.functional as F
from torch import nn

PAD_ID, BOS_ID, EOS_ID = 5, 3, 4
VOCAB_SIZE = 16000
MAX_LEN_IN, MAX_LEN_OUT = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

class SpaceTokenizer:
    def __init__(self):
        self.start_id = 6
        self.word2id = {}
        self.id2word = {}
    def encode(self, text, out_type=int):
        toks = str(text).strip().split()
        ids = []
        for w in toks:
            if w not in self.word2id:
                nid = self.start_id + len(self.word2id)
                if nid >= VOCAB_SIZE - 1:
                    nid = self.start_id
                self.word2id[w] = nid
                self.id2word[nid] = w
            ids.append(self.word2id[w])
        return ids
    def decode(self, ids):
        words = []
        for i in ids:
            if i in (PAD_ID, BOS_ID, EOS_ID):
                continue
            w = self.id2word.get(int(i), "")
            if w:
                words.append(w)
        return " ".join(words).strip()

try:
    import sentencepiece as spm
    if os.path.exists("spm.model"):
        sp = spm.SentencePieceProcessor(model_file="spm.model")
        USE_SPM = True
    else:
        sp = SpaceTokenizer()
        USE_SPM = False
except Exception:
    sp = SpaceTokenizer()
    USE_SPM = False

def sp_encode(s, out_type=int):
    return sp.encode(s, out_type=int) if USE_SPM else sp.encode(s, out_type=int)
def sp_decode(ids):
    return sp.decode(ids) if USE_SPM else sp.decode(ids)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, heads=4, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0
        self.h = heads
        self.d_k = d_model // heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None):
        B, Tq, C = q.shape
        Tk = k.shape[1]
        h, dk = self.h, self.d_k
        q = self.q(q).view(B, Tq, h, dk).transpose(1, 2)
        k = self.k(k).view(B, Tk, h, dk).transpose(1, 2)
        v = self.v(v).view(B, Tk, h, dk).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            mask = mask.to(scores.device)
            if mask.dim() == 4 and mask.size(-2) == 1:
                pass
            scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
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
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    def forward(self, x, src_mask):
        x_norm = self.ln1(x)
        x = x + self.self_attn(x_norm, x_norm, x_norm, src_mask)
        x = x + self.ffn(self.ln2(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, heads=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
    def forward(self, y, mem, tgt_mask, mem_mask):
        y_norm = self.ln1(y)
        y = y + self.self_attn(y_norm, y_norm, y_norm, tgt_mask)
        y = y + self.cross_attn(self.ln2(y), mem, mem, mem_mask)
        y = y + self.ffn(self.ln3(y))
        return y

def make_pad_mask(seq, pad=PAD_ID):
    return (seq == pad).unsqueeze(1).unsqueeze(2)
def make_causal_mask(T):
    return torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(0)

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
        src = src.to(DEVICE).long()
        tgt = tgt.to(DEVICE).long()
        src_mask = make_pad_mask(src).to(DEVICE)
        tgt_pad_mask = make_pad_mask(tgt).to(DEVICE)
        causal = make_causal_mask(tgt.size(1)).to(DEVICE)
        tgt_mask = tgt_pad_mask | causal
        mem = self.pos(self.embed(src))
        for layer in self.enc_layers:
            mem = layer(mem, src_mask)
        mem = self.ln_enc(mem)
        y = self.pos(self.embed(tgt))
        for layer in self.dec_layers:
            y = layer(y, mem, tgt_mask, src_mask)
        y = self.ln_dec(y)
        return self.out(y)

model = TransformerChat().to(DEVICE)
if os.path.exists("model.pt"):
    try:
        state = torch.load("model.pt", map_location=DEVICE)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        LOADED_WEIGHTS = True
        LOAD_NOTE = f"Loaded model.pt (missing: {len(missing)}, unexpected: {len(unexpected)})"
    except Exception as e:
        LOADED_WEIGHTS = False
        LOAD_NOTE = f"Failed to load model.pt, using random weights. Reason: {e}"
else:
    LOADED_WEIGHTS = False
    LOAD_NOTE = "model.pt not found — using random weights."
model.eval()

def encode_str(s, add_bos=False, add_eos=False, max_len=MAX_LEN_IN):
    ids = sp_encode(s, out_type=int)
    if add_bos:
        ids = [BOS_ID] + ids
    if add_eos:
        ids = ids + [EOS_ID]
    if not ids:
        ids = [BOS_ID, EOS_ID]
    return ids[:max_len]
def ids_to_text(ids):
    toks = []
    for t in ids:
        t = int(t)
        if t in (PAD_ID, BOS_ID):
            continue
        if t == EOS_ID:
            break
        toks.append(t)
    return sp_decode(toks)

@torch.no_grad()
def greedy_decode(src):
    ys = torch.tensor([[BOS_ID]], device=DEVICE, dtype=torch.long)
    for _ in range(MAX_LEN_OUT):
        logits = model(src, ys)
        next_id = logits[:, -1, :].argmax(-1, keepdim=True)
        ys = torch.cat([ys, next_id], dim=1)
        if int(next_id.item()) == EOS_ID:
            break
    return ys.squeeze(0).tolist()

@torch.no_grad()
def beam_decode(src, beam_width=3):
    beams = [(torch.tensor([[BOS_ID]], device=DEVICE, dtype=torch.long), 0.0)]
    for _ in range(MAX_LEN_OUT):
        new_beams = []
        for ys, score in beams:
            logits = model(src, ys)
            probs = F.log_softmax(logits[:, -1, :], dim=-1)
            topv, topi = torch.topk(probs, k=beam_width, dim=-1)
            for k in range(beam_width):
                next_tok = topi[0, k].view(1, 1)
                new_seq = torch.cat([ys, next_tok], dim=1)
                new_score = score + float(topv[0, k].item())
                new_beams.append((new_seq, new_score))
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]
        if all(int(seq[0, -1].item()) == EOS_ID for seq, _ in beams):
            break
    best_seq = beams[0][0].squeeze(0).tolist()
    return best_seq

def generate_reply(emotion, situation, customer, mode="Greedy"):
    prompt = f"emotion: {emotion} | situation: {situation} | customer: {customer} agent:"
    src_ids = encode_str(prompt, add_bos=True, add_eos=True, max_len=MAX_LEN_IN)
    src = torch.tensor([src_ids], device=DEVICE, dtype=torch.long)
    if mode == "Greedy":
        out_ids = greedy_decode(src)
    else:
        out_ids = beam_decode(src)
    return ids_to_text(out_ids)

st.title("Empathetic Chatbot (Transformer-from-Scratch)")
st.caption("Runs even if weights/tokenizer are missing (falls back safely).")
st.info(("Tokenizer: SentencePiece" if USE_SPM else "Tokenizer: Space-split fallback") + " • " + LOAD_NOTE)

emotion = st.text_input("Emotion", "grateful")
situation = st.text_input("Situation", "I passed my exam today!")
customer = st.text_area("Customer Message", "I'm so happy! I studied hard and finally passed.")
mode = st.selectbox("Choose Decoding Strategy", ["Greedy", "Beam Search"])

if st.button("Generate Reply"):
    with st.spinner("Generating reply..."):
        reply = generate_reply(emotion, situation, customer, mode)
    st.success(f"**{mode} Output:** {reply}")
