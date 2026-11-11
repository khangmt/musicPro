import json, torch, numpy as np
from transformers import AutoTokenizer, AutoModel
C=[c for c in json.load(open('../study1_listing/corpus_all.json')) if c['label'] in ('bolero','vpop','vpop?')]
tok=AutoTokenizer.from_pretrained('vinai/phobert-base-v2'); m=AutoModel.from_pretrained('vinai/phobert-base-v2').eval()
torch.set_num_threads(8)
out={}
for i,c in enumerate(C):
    lines=[l for l in dict.fromkeys(c['lyrics']) if l.strip()]
    text=' '.join(lines)
    ids=tok(text,add_special_tokens=False)['input_ids']
    chunks=[ids[j:j+254] for j in range(0,max(1,len(ids)),254)][:8]
    vecs=[]
    with torch.no_grad():
        for ch in chunks:
            x=torch.tensor([[tok.cls_token_id]+ch+[tok.sep_token_id]])
            h=m(input_ids=x).last_hidden_state[0,1:-1].mean(0); vecs.append(h.numpy())
    out[c['hac_id']]=np.mean(vecs,0).tolist()
    if i%100==0: print(i,len(C),flush=True)
json.dump(out,open('emb.json','w'))
