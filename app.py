
import streamlit as st
import torch, torch.nn.functional as F, math
from torch import nn
import sentencepiece as spm

PAD_ID, BOS_ID, EOS_ID = 5, 3, 4
VOCAB_SIZE = 16000
MAX_LEN_IN, MAX_LEN_OUT = 128, 64


class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=2048):
        super().__init__()
        pe=torch.zeros(max_len,d_model)
        pos=torch.arange(0,max_len).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2],pe[:,1::2]=torch.sin(pos*div),torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x): return x+self.pe[:,:x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model=256,heads=4,dropout=0.1):
        super().__init__()
        self.h, self.dk = heads, d_model//heads
        self.q,self.k,self.v,self.o = [nn.Linear(d_model,d_model) for _ in range(4)]
        self.drop=nn.Dropout(dropout)
    def forward(self,q,k,v,mask=None):
        B,Tq,_=q.shape; dk=self.dk; h=self.h
        q=self.q(q).view(B,Tq,h,dk).transpose(1,2)
        k=self.k(k).view(B,-1,h,dk).transpose(1,2)
        v=self.v(v).view(B,-1,h,dk).transpose(1,2)
        scores=(q@k.transpose(-2,-1))/math.sqrt(dk)
        if mask is not None: scores=scores.masked_fill(mask,float('-inf'))
        attn=self.drop(F.softmax(scores,dim=-1))
        out=(attn@v).transpose(1,2).contiguous().view(B,Tq,h*dk)
        return self.o(out)

class FFN(nn.Module):
    def __init__(self,d_model=256,d_ff=1024,dropout=0.1):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(d_model,d_ff),nn.ReLU(),nn.Dropout(dropout),
            nn.Linear(d_ff,d_model),nn.Dropout(dropout))
    def forward(self,x): return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self,d_model=256,heads=4,d_ff=1024,dropout=0.1):
        super().__init__()
        self.sa=MultiHeadAttention(d_model,heads,dropout)
        self.ff=FFN(d_model,d_ff,dropout)
        self.l1,self.l2=nn.LayerNorm(d_model),nn.LayerNorm(d_model)
    def forward(self,x,mask):
        x=x+self.sa(self.l1(x),self.l1(x),self.l1(x),mask)
        return x+self.ff(self.l2(x))

class DecoderLayer(nn.Module):
    def __init__(self,d_model=256,heads=4,d_ff=1024,dropout=0.1):
        super().__init__()
        self.sa=MultiHeadAttention(d_model,heads,dropout)
        self.ca=MultiHeadAttention(d_model,heads,dropout)
        self.ff=FFN(d_model,d_ff,dropout)
        self.l1,self.l2,self.l3=nn.LayerNorm(d_model),nn.LayerNorm(d_model),nn.LayerNorm(d_model)
    def forward(self,y,mem,tgt_mask,mem_mask):
        y=y+self.sa(self.l1(y),self.l1(y),self.l1(y),tgt_mask)
        y=y+self.ca(self.l2(y),mem,mem,mem_mask)
        return y+self.ff(self.l3(y))

def make_pad_mask(seq,pad=PAD_ID): return (seq==pad).unsqueeze(1).unsqueeze(2)
def make_causal_mask(sz): return torch.triu(torch.ones(sz,sz,dtype=torch.bool),1).unsqueeze(0).unsqueeze(0)

class TransformerChat(nn.Module):
    def __init__(self,vocab_size=VOCAB_SIZE,d_model=256,heads=4,n_enc=3,n_dec=3,d_ff=1024,dropout=0.1):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,d_model,padding_idx=PAD_ID)
        self.pos=PositionalEncoding(d_model)
        self.encs=nn.ModuleList([EncoderLayer(d_model,heads,d_ff,dropout) for _ in range(n_enc)])
        self.decs=nn.ModuleList([DecoderLayer(d_model,heads,d_ff,dropout) for _ in range(n_dec)])
        self.out=nn.Linear(d_model,vocab_size)
    def forward(self,src,tgt):
        spad,tpad=make_pad_mask(src),make_pad_mask(tgt)
        causal=make_causal_mask(tgt.size(1)); tmask=tpad|causal
        mem=self.pos(self.embed(src))
        for e in self.encs: mem=e(mem,spad)
        y=self.pos(self.embed(tgt))
        for d in self.decs: y=d(y,mem,tmask,spad)
        return self.out(y)

sp=spm.SentencePieceProcessor(model_file='spm.model')
model=TransformerChat(); model.load_state_dict(torch.load('model.pt',map_location='cpu')); model.eval()

def encode_str(s,add_bos=True,add_eos=True,max_len=MAX_LEN_IN):
    ids=sp.encode(s,out_type=int)
    if add_bos: ids=[BOS_ID]+ids
    if add_eos: ids=ids+[EOS_ID]
    return ids[:max_len]

def ids_to_text(ids):
    toks=[]
    for t in ids:
        if t in (PAD_ID,BOS_ID): continue
        if t==EOS_ID: break
        toks.append(int(t))
    return sp.decode(toks)

def generate_reply(emotion,situation,customer):
    x=f"emotion: {emotion} | situation: {situation} | customer: {customer} agent:"
    src=torch.tensor([encode_str(x)])
    ys=torch.tensor([[BOS_ID]])
    for _ in range(MAX_LEN_OUT):
        logits=model(src,ys)
        nxt=logits[:,-1,:].argmax(-1,keepdim=True)
        ys=torch.cat([ys,nxt],dim=1)
        if nxt.item()==EOS_ID: break
    return ids_to_text(ys.squeeze(0).tolist())

st.title("🤖 Empathetic Chatbot (Transformer-from-Scratch)")
emotion=st.text_input("Emotion","grateful")
situation=st.text_input("Situation","I passed my exam today!")
customer=st.text_area("Customer Message","I'm so happy! I studied hard and finally passed.")
if st.button("Generate Reply"):
    st.success(generate_reply(emotion,situation,customer))
