#!/usr/bin/env python
"""Initial ADNI MRI+MMSE RGAF/SHMR training pipeline. No trained results are included."""
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

LABEL_MAP = {"CN": 0, "MCI": 1, "AD": 2}
REQUIRED = ["PTID", "split", "diagnosis_label", "MMSE", "nifti_path", "approved_for_training"]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def truthy(x): return str(x).strip().lower() in {"true", "1", "yes", "y", "approved"}

def read_manifest(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing: raise ValueError(f"Missing required manifest columns: {missing}")
    df = df.copy()
    df["PTID"] = df.PTID.astype(str).str.strip()
    df["split"] = df.split.astype(str).str.strip().str.lower().replace({"val":"validation"})
    df["diagnosis_label"] = df.diagnosis_label.astype(str).str.strip().str.upper()
    df["label"] = df.diagnosis_label.map(LABEL_MAP)
    df["MMSE"] = pd.to_numeric(df.MMSE, errors="coerce")
    df["approved"] = df.approved_for_training.map(truthy)
    df["nifti_path"] = df.nifti_path.fillna("").astype(str).str.strip()
    if df.PTID.eq("").any(): raise ValueError("Blank PTID found.")
    if not set(df.split.unique()).issubset({"train", "validation", "test"}): raise ValueError("split must be train/validation/test")
    crossing = df.groupby("PTID").split.nunique()
    if (crossing > 1).any(): raise ValueError(f"Participant leakage across splits: {crossing[crossing>1].index[:10].tolist()}")
    if df.label.isna().any(): raise ValueError(f"Unknown diagnosis labels: {df.loc[df.label.isna(),'diagnosis_label'].unique().tolist()}")
    if df.MMSE.isna().any(): raise ValueError(f"{int(df.MMSE.isna().sum())} rows have missing MMSE")
    if (~df.approved).any(): raise ValueError(f"{int((~df.approved).sum())} rows are not explicitly approved_for_training=true. Complete source, conversion and visual QC first.")
    bad = df.nifti_path.eq("") | ~df.nifti_path.map(lambda p: Path(p).is_file())
    if bad.any(): raise ValueError(f"{int(bad.sum())} NIfTI paths missing/unavailable. Example rows: {df.loc[bad,['PTID','nifti_path']].head().to_dict('records')}")
    for s in ("train", "validation", "test"):
        if not (df.split == s).any(): raise ValueError(f"No rows in split={s}")
    print("Manifest checks passed")
    print("Rows/split:", df.split.value_counts().to_dict())
    print("Unique participants/split:", df.groupby('split').PTID.nunique().to_dict())
    print("Labels/split:", {s:df.loc[df.split==s,'diagnosis_label'].value_counts().to_dict() for s in ('train','validation','test')})
    return df

def load_mri(path, target=(64,64,64)):
    img = nib.as_closest_canonical(nib.load(path))
    a = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    if a.ndim != 3 or not np.isfinite(a).all(): raise ValueError(f"Invalid 3D volume/nonfinite data: {path}, shape={a.shape}")
    mask = np.abs(a) > 0
    if mask.sum() < 1000: raise ValueError(f"Near-empty MRI volume: {path}")
    lo, hi = np.percentile(a[mask], [1,99]); a = np.clip(a,lo,hi)
    vals = a[mask]; sd=float(vals.std())
    if not np.isfinite(sd) or sd < 1e-6: raise ValueError(f"Degenerate intensity distribution: {path}")
    a = (a-float(vals.mean()))/sd; a[~np.isfinite(a)] = 0
    t = torch.from_numpy(a.copy())[None,None]
    return F.interpolate(t, size=target, mode='trilinear', align_corners=False)[0].numpy().astype(np.float32)

class PairedADNI(Dataset):
    def __init__(self, df, mmse_mean, mmse_std): self.df=df.reset_index(drop=True); self.mean=mmse_mean; self.std=max(mmse_std,1e-6)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        return {"mri":torch.from_numpy(load_mri(r.nifti_path)), "clinical":torch.tensor([(float(r.MMSE)-self.mean)/self.std],dtype=torch.float32), "label":torch.tensor(int(r.label),dtype=torch.long), "ptid":str(r.PTID)}

class MRIEncoder(nn.Module):
    def __init__(self,out=64):
        super().__init__(); self.net=nn.Sequential(nn.Conv3d(1,8,3,2,1),nn.BatchNorm3d(8),nn.GELU(),nn.Conv3d(8,16,3,2,1),nn.BatchNorm3d(16),nn.GELU(),nn.Conv3d(16,32,3,2,1),nn.BatchNorm3d(32),nn.GELU(),nn.Conv3d(32,64,3,2,1),nn.BatchNorm3d(64),nn.GELU(),nn.AdaptiveAvgPool3d(1),nn.Flatten(),nn.Linear(64,out),nn.LayerNorm(out),nn.GELU())
    def forward(self,x): return self.net(x)
class ClinicalEncoder(nn.Module):
    def __init__(self,out=16):
        super().__init__(); self.net=nn.Sequential(nn.Linear(1,16),nn.GELU(),nn.Dropout(.1),nn.Linear(16,out),nn.LayerNorm(out),nn.GELU())
    def forward(self,x): return self.net(x)

class RGAF(nn.Module):
    def __init__(self,dims,fusion=64,hidden=64,drop=.1):
        super().__init__(); self.mods=list(dims)
        self.proj=nn.ModuleDict({m:nn.Sequential(nn.Linear(d,fusion),nn.LayerNorm(fusion),nn.GELU()) for m,d in dims.items()})
        self.gate=nn.ModuleDict({m:nn.Sequential(nn.Linear(fusion+2,hidden),nn.GELU(),nn.Dropout(drop),nn.Linear(hidden,1)) for m in dims})
        self.norm=nn.LayerNorm(fusion); self.drop=nn.Dropout(drop)
    def forward(self,reps,obs,rec):
        z=[]; scores=[]; valid=[]
        for m in self.mods:
            h=self.proj[m](reps[m]); z.append(h)
            scores.append(self.gate[m](torch.cat([h,obs[m].float().view(-1,1),rec[m].float().view(-1,1)],1)))
            valid.append(obs[m].bool()|rec[m].bool())
        valid=torch.stack(valid,1)
        if not valid.any(1).all(): raise ValueError("A sample has no usable modality")
        s=torch.cat(scores,1).masked_fill(~valid,torch.finfo(z[0].dtype).min)
        w=torch.softmax(s,1); fused=(torch.stack(z,1)*w.unsqueeze(-1)).sum(1)
        return self.norm(self.drop(fused)),w
class SHMR(nn.Module):
    def __init__(self,dims,hidden=64,drop=.1):
        super().__init__(); self.mods=list(dims); self.net=nn.ModuleDict()
        for target in self.mods:
            inp=sum(d for m,d in dims.items() if m!=target)+len(dims)-1
            self.net[target]=nn.Sequential(nn.Linear(inp,hidden),nn.GELU(),nn.Dropout(drop),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,dims[target]))
    def forward(self,x,avail):
        out={}
        for target in self.mods:
            xs=[]; flags=[]
            for src in self.mods:
                if src==target: continue
                f=avail[src].float().view(-1,1); xs.append(x[src]*f); flags.append(f)
            out[target]=self.net[target](torch.cat(xs+flags,1))
        return out
class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.mri=MRIEncoder(64); self.clin=ClinicalEncoder(16); dims={'mri':64,'clinical':16}; self.shmr=SHMR(dims); self.rgaf=RGAF(dims); self.cls=nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,3)); self.mods=list(dims)
    def encode(self,b): return {'mri':self.mri(b['mri']),'clinical':self.clin(b['clinical'])}
    def forward(self,b,avail=None):
        x=self.encode(b); n=x['mri'].shape[0]
        if avail is None: avail={m:torch.ones(n,dtype=torch.bool,device=x['mri'].device) for m in self.mods}
        rec=self.shmr(x,avail); reps={m:torch.where(avail[m].view(-1,1),x[m],rec[m]) for m in self.mods}; flags={m:~avail[m].bool() for m in self.mods}
        fused,w=self.rgaf(reps,avail,flags)
        return {'logits':self.cls(fused),'weights':w,'recovered':rec,'embeddings':x}
    def objective(self,b,y,mask_prob=.3,recovery_weight=.2):
        x=self.encode(b); n=y.shape[0]; clean={m:torch.ones(n,dtype=torch.bool,device=y.device) for m in self.mods}; corrupted={m:v.clone() for m,v in clean.items()}
        for i in range(n):
            if torch.rand((),device=y.device).item()<mask_prob:
                corrupted['mri' if torch.rand((),device=y.device).item()<.5 else 'clinical'][i]=False
        rec=self.shmr(x,corrupted); reps={m:torch.where(corrupted[m].view(-1,1),x[m],rec[m]) for m in self.mods}; flags={m:~corrupted[m] for m in self.mods}
        fused,w=self.rgaf(reps,corrupted,flags); logits=self.cls(fused); terms=[]
        for m in self.mods:
            target=clean[m]&~corrupted[m]
            if target.any(): terms.append(F.mse_loss(rec[m][target],x[m].detach()[target]))
        rloss=torch.stack(terms).mean() if terms else logits.new_zeros(()); closs=F.cross_entropy(logits,y); return closs+recovery_weight*rloss, closs.detach(), rloss.detach()

def device_batch(b,dev): return {k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in b.items()}
@torch.no_grad()
def evaluate(model,loader,dev):
    model.eval(); ys=[]; ps=[]; ls=[]; ws=[]
    for b0 in loader:
        b=device_batch(b0,dev); o=model(b); ls.append(float(F.cross_entropy(o['logits'],b['label']).cpu())); ys.extend(b['label'].cpu().tolist()); ps.extend(o['logits'].argmax(1).cpu().tolist()); ws.extend(o['weights'].cpu().tolist())
    cm=np.zeros((3,3),dtype=int)
    for y,p in zip(ys,ps): cm[y,p]+=1
    f1=[]; recalls=[]
    for c in range(3):
        tp=int(((np.array(ys)==c)&(np.array(ps)==c)).sum()); fp=int(((np.array(ys)!=c)&(np.array(ps)==c)).sum()); fn=int(((np.array(ys)==c)&(np.array(ps)!=c)).sum()); prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0; recalls.append(rec); f1.append(2*prec*rec/(prec+rec) if prec+rec else 0)
    return {'n':len(ys),'accuracy':float(np.mean(np.array(ys)==np.array(ps))),'macro_f1':float(np.mean(f1)),'balanced_accuracy':float(np.mean(recalls)),'confusion_matrix_true_rows_pred_cols':cm.tolist(),'cross_entropy':float(np.mean(ls)),'mean_fusion_weights_mri_clinical':np.mean(ws,axis=0).tolist()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--outdir',required=True); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--batch-size',type=int,default=2); ap.add_argument('--lr',type=float,default=1e-4); ap.add_argument('--mask-prob',type=float,default=.3); ap.add_argument('--recovery-weight',type=float,default=.2); ap.add_argument('--patience',type=int,default=6); ap.add_argument('--seed',type=int,default=20261019); args=ap.parse_args()
    seed_all(args.seed); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True); df=read_manifest(args.manifest)
    tr=df[df.split=='train']; va=df[df.split=='validation']; te=df[df.split=='test']
    mean=float(tr.MMSE.mean()); std=float(tr.MMSE.std(ddof=0))
    if not np.isfinite(std) or std<1e-6: raise ValueError('Training MMSE standard deviation invalid/zero')
    loaders=[]
    for part,shuffle in [(tr,True),(va,False),(te,False)]: loaders.append(DataLoader(PairedADNI(part,mean,std),batch_size=args.batch_size,shuffle=shuffle,num_workers=0))
    train_loader,val_loader,test_loader=loaders; dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=Model().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    best=float('inf'); best_ep=0; stale=0; hist=[]
    print(f'Device={dev}; rows train/validation/test={len(tr)}/{len(va)}/{len(te)}; MMSE standardizer fitted on train only')
    for ep in range(1,args.epochs+1):
        model.train(); tl=[]; cl=[]; rl=[]
        for b0 in train_loader:
            b=device_batch(b0,dev); opt.zero_grad(set_to_none=True); loss,c,r=model.objective(b,b['label'],args.mask_prob,args.recovery_weight)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); tl.append(float(loss.detach().cpu())); cl.append(float(c.cpu())); rl.append(float(r.cpu()))
        vm=evaluate(model,val_loader,dev); row={'epoch':ep,'train_loss':float(np.mean(tl)),'train_classification_loss':float(np.mean(cl)),'train_recovery_loss':float(np.mean(rl)),'val_metrics':vm}; hist.append(row); print(json.dumps(row))
        if vm['cross_entropy']<best: best=vm['cross_entropy']; best_ep=ep; stale=0; torch.save({'model_state_dict':model.state_dict(),'epoch':ep,'mmse_mean_train':mean,'mmse_std_train':std,'label_map':LABEL_MAP,'seed':args.seed},out/'best_model.pt')
        else:
            stale+=1
            if stale>=args.patience: print(f'Early stop; best epoch {best_ep}'); break
    pd.DataFrame(hist).to_csv(out/'training_history.csv',index=False)
    ck=torch.load(out/'best_model.pt',map_location=dev,weights_only=False); model.load_state_dict(ck['model_state_dict']); test=evaluate(model,test_loader,dev)
    report={'experiment':'ADNI MRI+MMSE RGAF+SHMR prototype','best_epoch':best_ep,'best_validation_cross_entropy':best,'test_metrics':test,'test_used_for_checkpoint_selection':False,'clinical_features':['MMSE'],'clinical_standardization_fit_on_train_only':True,'participant_split_check_passed':True,'mri_preprocessing':'canonical orientation; per-volume nonzero percentile clipping and z-score; trilinear resize to 64^3','label_map':LABEL_MAP,'args':vars(args),'rows_by_split':df.split.value_counts().to_dict(),'participants_by_split':df.groupby('split').PTID.nunique().to_dict(),'limitations':['Prototype single run; not five-fold results','MMSE may be strongly associated with diagnosis; compare MRI-only and clinical-only baselines','Fixed-size resizing requires documented QC','RGAF weights are learned gates, not calibrated uncertainty','Clinical validity is not established']}
    (out/'evaluation_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print('FINAL TEST REPORT'); print(json.dumps(report,indent=2)); print('Artifacts:',out)
if __name__=='__main__': main()
