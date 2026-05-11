"""Rerun only checks 1, 8, 9 from test_full_pipeline.py."""
import os, sys, numpy as np, torch
from basicsr.archs.iaml_arch import IAMLNet
from basicsr.losses.iaml_loss import TotalLoss
from basicsr.metrics import calculate_psnr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = IAMLNet().to(device)

results = {}

# ── Check 1 ──────────────────────────────────────────────────────────────────
try:
    model.train()
    lq = torch.rand(8, 3, 256, 256, device=device)
    gt = torch.rand(8, 3, 256, 256, device=device)
    enhanced, pairs = model(lq, gt)
    expected_pairs = [(8,256,16,16),(8,128,32,32),(8,64,64,64),(8,32,128,128)]
    for i,(u,t) in enumerate(pairs): print(f"  pairs[{i}][0]={tuple(u.shape)}")
    fails=[f"pairs[{i}][0]={tuple(pairs[i][0].shape)} exp={expected_pairs[i]}"
           for i in range(4) if tuple(pairs[i][0].shape)!=expected_pairs[i]]
    if not (enhanced.shape==torch.Size([8,3,256,256]) and enhanced.min()>=0 and enhanced.max()<=1):
        fails.append("enhanced shape/range")
    results[1]=('FAIL','; '.join(fails)) if fails else ('PASS',f'all shapes match')
except Exception as e: results[1]=('FAIL',str(e))

# ── Check 8 ──────────────────────────────────────────────────────────────────
def padded_size(H,W,m=32): return ((H+m-1)//m)*m,((W+m-1)//m)*m
try:
    cases=[(1,3,600,400),(1,3,601,401),(1,3,720,1280)]
    expected_padded=[(608,416),(608,416),(736,1280)]
    model.eval(); fails=[]
    for (B,C,H,W),(pH,pW) in zip(cases,expected_padded):
        out=model.inference(torch.rand(B,C,H,W,device=device))
        cpH,cpW=padded_size(H,W)
        print(f"  ({H}×{W})→pad({cpH}×{cpW})→out{tuple(out.shape[2:])}")
        if out.shape!=torch.Size([B,C,H,W]): fails.append(f"({H}×{W}) out={tuple(out.shape)}")
        if (cpH,cpW)!=(pH,pW): fails.append(f"({H}×{W}) padded ({cpH}×{cpW}) exp ({pH}×{pW})")
    results[8]=('FAIL','; '.join(fails)) if fails else ('PASS','all shapes match')
except Exception as e: results[8]=('FAIL',str(e))

# ── Check 9 ──────────────────────────────────────────────────────────────────
try:
    img_a=(np.random.rand(400,600,3)*255).astype(np.float32)
    psnr_id=calculate_psnr(img_a,img_a,crop_border=0,test_y_channel=True)
    img_z=np.zeros((400,600,3),dtype=np.float32)
    img_o=np.full((400,600,3),255.,dtype=np.float32)
    psnr_diff=calculate_psnr(img_z,img_o,crop_border=0,test_y_channel=True)
    noise=(np.random.rand(400,600,3)*10).astype(np.float32)
    img_b=np.clip(img_a+noise,0,255)
    psnr_real=calculate_psnr(img_a,img_b,crop_border=0,test_y_channel=True)
    print(f"  PSNR identical={psnr_id:.2f}  diff={psnr_diff:.2f}  realistic={psnr_real:.2f}")
    fails=[]
    if psnr_id<=60: fails.append(f"identical={psnr_id:.2f} not >60")
    if not (20<psnr_real<60): fails.append(f"realistic={psnr_real:.2f} not in (20,60)")
    results[9]=('FAIL','; '.join(fails)) if fails else ('PASS',f'id={psnr_id:.1f} real={psnr_real:.1f} dB')
except Exception as e: results[9]=('FAIL',str(e))

# ── Summary ───────────────────────────────────────────────────────────────────
for i in [1,8,9]:
    s,d=results[i]; print(f"Check {i}: {s} — {d}")
if all(results[i][0]=='PASS' for i in [1,8,9]): print("All 3 re-checks PASS")
else: sys.exit(1)
