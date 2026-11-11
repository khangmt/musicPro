import json
src=open('r3_text.py').read(); exec(src[:src.index("out = {")])
cp=set()
for h,(g,L) in songs.items():
    for i in range(len(L)-1):
        if len(L[i])==6 and len(L[i+1])==8: cp.add((h,i)); cp.add((h,i+1))
D['couplet']=[int((h,i) in cp) for h,i in zip(D.song,D.line_idx)]
d2 = D[D.genre.isin(['bolero','vpop'])].copy(); d2['bolero']=(d2.genre=='bolero').astype(int)
def gee(df,f,group='song'):
    m=smf.gee(f,group,df,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Exchangeable()).fit(); ci=m.conf_int()
    return {k:dict(OR=round(float(np.exp(m.params[k])),2),lo=round(float(np.exp(ci.loc[k,0])),2),hi=round(float(np.exp(ci.loc[k,1])),2),p=float(m.pvalues[k])) for k in m.params.index if 'change' in k}
F='lv ~ change * bolero + isfinal + isfirst + relpos + line_len + log_seg_len + minor + rel_line + song_final_line'
out={'share_syll_in_couplets':d2.groupby('genre').couplet.mean().round(3).to_dict(),
     'non_couplet':gee(d2[d2.couplet==0],F),
     'non_6_or_8_lines':gee(d2[~d2.line_len.isin([6,8])],F)}
for g in ['bolero','vpop']:
    x=d2[(d2.genre==g)&(d2.couplet==0)]
    out[g+'_noncouplet_delta']=round(float(x[x.change==1].lv.mean()-x[x.change==0].lv.mean()),3)
    x=d2[(d2.genre==g)&(~d2.line_len.isin([6,8]))]
    out[g+'_non68_delta']=round(float(x[x.change==1].lv.mean()-x[x.change==0].lv.mean()),3)
json.dump(out,open('r3b.json','w'),indent=1); print(json.dumps(out,indent=1))
