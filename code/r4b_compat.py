"""R4b: vocal-chord compatibility GEE with tone x event x genre interactions (88 recordings)."""
import pandas as pd, numpy as np, json, statsmodels.api as sm, statsmodels.formula.api as smf
A=pd.read_csv('r4_audio_syllables.csv'); A['bolero']=(A.genre=='bolero').astype(int); A['log_dur']=np.log(A.dur.clip(lower=.03))
c=A.dropna(subset=['comp']); out={}
for k,f in {'change3':'comp ~ lv*sheet_change*bolero + log_dur + isfinal','final3':'comp ~ lv*isfinal*bolero + sheet_change + log_dur'}.items():
    m=smf.gee(f,'song',c,cov_struct=sm.cov_struct.Exchangeable()).fit(); ci=m.conf_int()
    out[k]={n:dict(b=round(float(m.params[n]),3),lo=round(float(ci.loc[n,0]),3),hi=round(float(ci.loc[n,1]),3),p=float(m.pvalues[n])) for n in m.params.index}
json.dump(out,open('r4b_compat.json','w'),indent=1)
print(out['change3']['lv:sheet_change:bolero'], out['final3']['lv:isfinal:bolero'])
