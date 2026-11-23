#!/usr/bin/env python3
"""AMI Free Bio — name-enriched SentencePiece UNIGRAM, learned from official MSM data only.

A name-only conditional GRU generator (no elements, no islands, no templates).
Heldout species do NOT participate in tokenizer training (even as name-only corpus).
Run `train`, `generate`, or `final-train` from the project directory.
"""
import argparse, collections, json, math, os, random, subprocess, tempfile, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import sentencepiece as spm

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'bios_free.jsonl'
CANON=HERE/'monsters.json'
PAD,BOS,EOS,UNK=3,1,2,0

def source_rows():
    if not SOURCE.exists(): raise FileNotFoundError('Copy bios_free.jsonl from AMI Bio project to this directory first.')
    return [json.loads(line) for line in SOURCE.read_text(encoding='utf8').splitlines()]

def split(xs,seed=42):
    names=sorted({r['base_name'] for r in xs});random.Random(seed).shuffle(names)
    excluded=set(names[:round(len(names)*.15)])
    return [r for r in xs if r['base_name'] not in excluded], [r for r in xs if r['base_name'] in excluded],sorted(excluded)

def train_tokenizer(train_rows,out_prefix,vocab_size=1280,include_canonical=False):
    names=sorted({r['name'] for r in train_rows})
    if include_canonical and CANON.exists():
        names=sorted(set(names) | {r['name'] for r in json.loads(CANON.read_text())})
    # Short names and ordinary English words are not forced to be atomic.
    atomic_names=[n for n in names if len(n)>=4 and ' ' not in n and '\n' not in n and not n.startswith('<')]
    # A few repeats of true species names are tokenizer examples, NOT synthetic biographies.
    # The corpus contains train bios only, and no text from heldout monsters.
    corpus=out_prefix.parent/'tokenizer_corpus_internal.txt'
    with corpus.open('w',encoding='utf8') as f:
        for r in train_rows: f.write(r['bio'].replace('\n',' ')+'\n')
        for n in names:
            for _ in range(3):f.write(n+'\n')
    spm.SentencePieceTrainer.train(input=str(corpus),model_prefix=str(out_prefix),
        vocab_size=vocab_size,model_type='unigram',byte_fallback=True,hard_vocab_limit=False,
        user_defined_symbols=atomic_names,unk_id=UNK,bos_id=BOS,eos_id=EOS,pad_id=PAD,
        normalization_rule_name='identity',character_coverage=1.0,
        max_sentence_length=6000,num_threads=4)
    tok=spm.SentencePieceProcessor(model_file=str(out_prefix)+'.model')
    assert tok.pad_id()==PAD and tok.bos_id()==BOS and tok.eos_id()==EOS
    assert tok.decode(tok.encode('Bumbleler can sing.'))=='Bumbleler can sing.'
    corpus.unlink() # Do not redistribute copyrighted source texts twice.
    return tok, {'train_corpus_biographies':len(train_rows), 'name_strings_in_tokenizer_corpus':len(names),
            'atomic_named_monster_tokens':sum(tok.piece_to_id(n)!=tok.unk_id() and tok.id_to_piece(tok.piece_to_id(n))==n for n in atomic_names),
            'vocab_size':tok.vocab_size(),'tokenizer_algorithm':'SentencePiece Unigram + byte fallback + atomic known names',
            'actual_atomic_names':len(atomic_names)}

class FreeBio(nn.Module):
    def __init__(self,vocab,embedding=72,hidden=144,dropout=.15):
        super().__init__()
        self.token_emb=nn.Embedding(vocab,embedding,padding_idx=PAD)
        self.name_proj=nn.Sequential(nn.Linear(embedding,hidden),nn.Tanh())
        self.gru=nn.GRU(embedding+hidden,hidden,batch_first=True)
        self.head=nn.Linear(hidden,vocab)
        self.dropout=nn.Dropout(dropout)
        self.config={'vocab':vocab,'embedding':embedding,'hidden':hidden,'dropout':dropout}
    def condition(self,names):
        mask=(names!=PAD).unsqueeze(-1)
        z=self.token_emb(names)
        return self.name_proj((z*mask).sum(dim=1)/mask.sum(dim=1).clamp(min=1))
    def forward(self,names,input_ids):
        cond=self.condition(names)
        x=torch.cat((self.dropout(self.token_emb(input_ids)),cond[:,None,:].expand(-1,input_ids.shape[1],-1)),dim=-1)
        out,_=self.gru(x,cond.unsqueeze(0))
        return self.head(self.dropout(out))
    @torch.no_grad()
    def generate(self,name_ids,seed=9,temperature=.82,top_k=32,max_tokens=450):
        self.eval();rng=torch.Generator().manual_seed(seed)
        cond=self.condition(name_ids)
        hidden=cond[None,:,:].contiguous()
        inp=torch.tensor([[BOS]]);out=[];reason='max_tokens'
        for j in range(max_tokens):
            x=torch.cat((self.token_emb(inp),cond[:,None,:]),dim=-1)
            z,hidden=self.gru(x,hidden)
            logits=self.head(z[:,-1,:])/max(temperature,.01)
            logits[:,[PAD,BOS,UNK]]=float('-inf')
            if top_k and top_k < logits.shape[-1]:
                val,idx=logits.topk(top_k,dim=-1)
                p=F.softmax(val,dim=-1)
                inp=idx.gather(1,torch.multinomial(p,1,generator=rng))
            else: inp=torch.multinomial(F.softmax(logits,dim=-1),1,generator=rng)
            new=int(inp.item())
            if new==EOS:reason='end_of_bio_token';break
            out.append(new)
        return out,reason

def encode_rows(rows,tokenizer):
    return [{'name':r['name'],'n':tokenizer.encode(r['name']),
             'seq':[BOS]+tokenizer.encode(r['bio'])+[EOS],
             'bytes':len(r['bio'].encode('utf-8'))} for r in rows]

def make_batch(data):
    B=len(data);max_n=max(len(r['n']) for r in data);max_t=max(len(r['seq'])-1 for r in data)
    names=torch.full((B,max_n),PAD,dtype=torch.long)
    inputs=torch.full((B,max_t),PAD,dtype=torch.long)
    targets=torch.full((B,max_t),PAD,dtype=torch.long)
    for j,r in enumerate(data):
        names[j,:len(r['n'])]=torch.tensor(r['n'])
        inputs[j,:len(r['seq'])-1]=torch.tensor(r['seq'][:-1])
        targets[j,:len(r['seq'])-1]=torch.tensor(r['seq'][1:])
    return names,inputs,targets

def windows(rows,batch=14,max_len=144):
    out=[]
    for _ in range(batch):
        row=random.choice(rows)
        if len(row['seq'])<=max_len+1:segment=row['seq']
        else:
            begin=len(row['seq'])-max_len-1 if random.random()<.5 else random.randint(0,len(row['seq'])-max_len-1)
            segment=row['seq'][begin:begin+max_len+1]
        out.append({'n':row['n'],'seq':segment})
    return make_batch(out)

@torch.no_grad()
def evaluate(model,rows,wrong_names=False):
    model.eval();sum_loss=0;tot_tokens=0;tot_bytes=0;eos_probs=[]
    rows=sorted(rows,key=lambda x:len(x['seq']))
    for j in range(0,len(rows),6):
        chunk=rows[j:j+6];n,x,y=make_batch(chunk)
        if wrong_names and len(chunk)>1:n=n.roll(1,dims=0)
        logits=model(n,x)
        sum_loss+=float(F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=PAD,reduction='sum'))
        tot_tokens+=int((y!=PAD).sum());tot_bytes+=sum(r['bytes'] for r in chunk)
        for k,row in enumerate(chunk):eos_probs.append(float(logits[k,len(row['seq'])-2].softmax(-1)[EOS]))
    return {'loss_nats_per_token':sum_loss/tot_tokens,'bits_per_byte':sum_loss/(tot_bytes*math.log(2)),
            'avg_true_eos_probability':float(np.mean(eos_probs)), 'total_bio_bytes':tot_bytes,'total_target_tokens':tot_tokens}

def sample(model,tok,name,seed=9,temperature=.82,max_tokens=450):
    torch.set_num_threads(2)
    nameids=tok.encode(name)
    ids,why=model.generate(torch.tensor([nameids],dtype=torch.long),seed=seed,temperature=temperature,max_tokens=max_tokens)
    return {'name':name,'seed':seed,'stop_reason':why,'tokens':len(ids),
            'bio':tok.decode(ids).strip()}

def load(model_dir):
    folder=Path(model_dir);tok=spm.SentencePieceProcessor(model_file=str(folder/'tokenizer.model'))
    ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    model=FreeBio(**ck['config']);model.load_state_dict(ck['weights']);model.eval()
    return model,tok,ck

def train(model_dir,steps=1400,seed=20261122,full=False,eval_every=100):
    model_dir=Path(model_dir);model_dir.mkdir(parents=True,exist_ok=True)
    os.environ['OMP_NUM_THREADS']='4';torch.set_num_threads(4)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    all_rows=source_rows()
    if full:tr=all_rows;val=[];withheld=[]
    else:tr,val,withheld=split(all_rows)
    tic=time.perf_counter()
    tok,tok_info=train_tokenizer(tr,model_dir/'tokenizer',include_canonical=full)
    tr_encoded=encode_rows(tr,tok);val_encoded=encode_rows(val,tok)
    net=FreeBio(tok.vocab_size());nparams=sum(p.numel() for p in net.parameters())
    opt=torch.optim.AdamW(net.parameters(),lr=.0008,weight_decay=.02)
    best=float('inf');best_step=0;bad=0;history=[]
    last_score=None
    for step in range(1,steps+1):
        net.train();n,x,y=windows(tr_encoded)
        logits=net(n,x)
        loss=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=PAD)
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step()
        if step==1 or step%eval_every==0 or step==steps:
            score=evaluate(net,val_encoded) if val_encoded else {'bits_per_byte':float(loss.item())}
            last_score=score
            item={'step':step,'training_window_loss':round(float(loss),5),
                  'heldout_bits_per_byte':round(score['bits_per_byte'],5),
                  'elapsed_seconds':round(time.perf_counter()-tic,1)}
            history.append(item);print('FULL' if full else 'HOLDOUT',json.dumps(item),flush=True)
            if score['bits_per_byte']<best-.001:
                best=score['bits_per_byte'];best_step=step;bad=0
                torch.save({'weights':{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
                    'config':net.config,'seed':seed,'step':step,'model':'AMI Free Bio Unigram Named',
                    'training_only_official_bios':True},model_dir/'best.pt')
            else:bad+=1
            if not full and bad>=5 and step>=500:break
    net,tok,ck=load(model_dir)
    val_metric=evaluate(net,val_encoded) if val_encoded else None
    wrong=evaluate(net,val_encoded,wrong_names=True) if val_encoded else None
    demo=[]
    for name in ['Bumbleler','Gadla','Furcorn','Ghazt','Binbb','Shugabot']:
        for sd in [9,28,43]:demo.append(sample(net,tok,name,seed=sd))
    (model_dir/'samples.json').write_text(json.dumps(demo,indent=2,ensure_ascii=False))
    report={'kind':'free','tokenizer':tok_info,'official_training_bios':len(tr),
        'heldout_bios':len(val),'heldout_species':len(withheld),'parameter_count':nparams,
        'training_steps_best':best_step,'total_steps':history[-1]['step'],
        'validation':val_metric,'validation_with_wrong_names':wrong,
        'name_conditioning_advantage_nats':(wrong['loss_nats_per_token']-val_metric['loss_nats_per_token']) if wrong else None,
        'natural_stop_count':sum(d['stop_reason']=='end_of_bio_token' for d in demo),
        'sample_count':len(demo),'name_memo':'Names in tokenizer and source only; no user-supplied word lists or generated training bios.',
        'train_history':history,'total_seconds':round(time.perf_counter()-tic,2)}
    (model_dir/'report.json').write_text(json.dumps(report,indent=2))
    (model_dir/'name_tokens.json').write_text(json.dumps({n:tok.encode(n,out_type=str) for n in ['Bumbleler','Gadla','Furcorn','Rare Tweedle','Ghazt','Sol-stice']},indent=2,ensure_ascii=False))
    print('FINAL REPORT',json.dumps({k:v for k,v in report.items() if k!='train_history'},indent=2),flush=True)
    return report

def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='action',required=True)
    tr=sub.add_parser('train');tr.add_argument('--steps',type=int,default=1400);tr.add_argument('--out',default='heldout_model')
    fn=sub.add_parser('final-train');fn.add_argument('--steps',type=int,default=900);fn.add_argument('--out',default='complete_model')
    ge=sub.add_parser('generate');ge.add_argument('name');ge.add_argument('--dir',default='complete_model');ge.add_argument('--seed',type=int,default=9);ge.add_argument('--temperature',type=float,default=.82);ge.add_argument('--max-tokens',type=int,default=450)
    args=parser.parse_args()
    if args.action in ('train','final-train'):
        train(HERE/args.out,steps=args.steps,full=args.action=='final-train')
    else:
        net,tok,ck=load(HERE/args.dir)
        print(json.dumps(sample(net,tok,args.name,args.seed,args.temperature,args.max_tokens),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
