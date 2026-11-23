"""Lightweight name-conditioned recurrent biography generator."""
import torch
from torch import nn
import torch.nn.functional as F

class BioGRU(nn.Module):
    def __init__(self,vocab_size,emb_dim=96,hidden_dim=160,dropout=0.15):
        super().__init__()
        self.emb=nn.Embedding(vocab_size,emb_dim,padding_idx=3)
        self.name_init=nn.Sequential(nn.Linear(emb_dim,hidden_dim),nn.Tanh())
        self.gru=nn.GRU(emb_dim+hidden_dim,hidden_dim,num_layers=1,batch_first=True)
        self.out=nn.Linear(hidden_dim,vocab_size)
        self.dropout=nn.Dropout(dropout)
        self.hidden_dim=hidden_dim
        self.emb_dim=emb_dim
        self.dropout_rate=dropout
        self.vocab_size=vocab_size

    def name_condition(self, name_ids):
        mask=(name_ids!=3).float().unsqueeze(-1)
        v=(self.emb(name_ids)*mask).sum(1)/mask.sum(1).clamp(min=1)
        return self.name_init(v)

    def forward(self,name_ids, bio_in,hidden=None):
        cond=self.name_condition(name_ids)
        if hidden is None: hidden=cond.unsqueeze(0)
        emb=self.dropout(self.emb(bio_in))
        x=torch.cat((emb,cond.unsqueeze(1).expand(-1,emb.size(1),-1)),dim=-1)
        x,h=self.gru(x,hidden)
        return self.out(self.dropout(x)),h

    @torch.no_grad()
    def generate(self, name_ids, bos_id=1,eos_id=2,max_tokens=175,temperature=0.83,top_k=25,seed=1):
        self.eval(); rng=torch.Generator(device='cpu');rng.manual_seed(seed)
        cond=self.name_condition(name_ids)
        h=cond.unsqueeze(0)
        token=torch.full((name_ids.size(0),1),bos_id,dtype=torch.long)
        result=[]
        for _ in range(max_tokens):
            x=torch.cat((self.emb(token),cond.unsqueeze(1)),dim=-1)
            z,h=self.gru(x,h)
            logits=self.out(z[:,-1])/max(0.05,temperature)
            if top_k is not None and 0<top_k<logits.shape[-1]:
                values,ix=logits.topk(top_k,dim=-1)
                probs=F.softmax(values,dim=-1)
                next_id=ix.gather(1,torch.multinomial(probs,1,generator=rng))
            else:
                probs=F.softmax(logits,dim=-1)
                next_id=torch.multinomial(probs,1,generator=rng)
            nxt=next_id.item();
            if nxt==eos_id:break
            result.append(nxt)
            token=next_id
        return result
