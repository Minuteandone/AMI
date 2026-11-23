#!/usr/bin/env python3
"""Generate until the TRUE end-of-bio token; max_tokens is an emergency cap."""
import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F
import sentencepiece as spm
from model import BioGRU
ROOT=Path(__file__).resolve().parent

@torch.inference_mode()
def generate_bio(name,variant='template',seed=9,temperature=.82,top_k=25,max_tokens=1024,checkpoint=None):
    torch.set_num_threads(1)
    sp=spm.SentencePieceProcessor(model_file=str(ROOT/'msm_bio_sp.model'))
    path=Path(checkpoint) if checkpoint else ROOT/variant/'best.pt'
    ckpt=torch.load(path,weights_only=False,map_location='cpu')
    model=BioGRU(**ckpt['config']);model.load_state_dict(ckpt['weights']);model.eval()
    ids=sp.encode(name)[:32]
    n=torch.tensor([ids]);cond=model.name_condition(n)
    h=cond.unsqueeze(0);token=torch.tensor([[sp.bos_id()]])
    rng=torch.Generator();rng.manual_seed(seed)
    out=[];reason='max_tokens';eos_id=sp.eos_id()
    for _ in range(max_tokens):
        x=torch.cat((model.emb(token),cond.unsqueeze(1)),dim=-1)
        z,h=model.gru(x,h)
        logits=model.out(z[:,-1])/max(.05,temperature)
        logits[:,sp.pad_id()]=float('-inf')
        logits[:,sp.bos_id()]=float('-inf')
        if top_k and top_k<logits.shape[-1]:
            values,ix=logits.topk(top_k,dim=-1)
            next_tok=ix.gather(1,torch.multinomial(F.softmax(values,dim=-1),1,generator=rng))
        else:next_tok=torch.multinomial(F.softmax(logits,dim=-1),1,generator=rng)
        nxt=int(next_tok.item())
        if nxt==eos_id:
            reason='end_of_bio_token';break
        out.append(nxt);token=next_tok
    raw=sp.decode(out).strip()
    return {'name':name,'model':variant,'bio':raw.replace('[monster name]',name) if variant=='template' else raw,
            'raw_bio':raw,'stop_reason':reason,'ended_naturally':reason=='end_of_bio_token',
            'generated_tokens':len(out),'end_token':'</s> (END_BIO)',
            'max_tokens_safety_limit':max_tokens,'seed':seed,
            'checkpoint_step':ckpt.get('step')}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--model',default='template',choices=['template','free'])
    p.add_argument('--seed',type=int,default=9);p.add_argument('--temperature',type=float,default=.82)
    p.add_argument('--top-k',type=int,default=25);p.add_argument('--max-tokens',type=int,default=1024)
    p.add_argument('--checkpoint',default=None);a=p.parse_args()
    print(json.dumps(generate_bio(a.name,a.model,a.seed,a.temperature,a.top_k,a.max_tokens,a.checkpoint),indent=2,ensure_ascii=False))
