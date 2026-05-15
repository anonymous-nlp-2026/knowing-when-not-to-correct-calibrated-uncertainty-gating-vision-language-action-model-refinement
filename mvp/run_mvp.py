"""
3D Point-Reaching MVP: per-DOF uncertainty vs scalar uncertainty vs none.
Systematic bias scales with per-DOF sigma (bias = sigma * alpha).
Observation = noisy_action only (model does NOT see true target).
"""
import torch
import torch.nn as nn
import numpy as np

DEVICE = torch.device("cuda:0")
SEED = 42
# alpha > 1 ensures per-DOF sigma explains meaningful correction variance
ALPHA = 2.5


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_data(n, base_sx, base_sy, base_sz, use_estimated=False):
    """
    Per-sample independent noise scaling per axis.
    Perturbation = sigma * ALPHA (systematic bias scales with uncertainty).
    """
    targets = np.random.uniform(-1, 1, (n, 3))
    noise_scales = np.random.uniform(0.5, 2.0, (n, 3))
    base = np.array([base_sx, base_sy, base_sz])
    sigmas = base * noise_scales
    perturbation = sigmas * ALPHA
    noise = np.random.randn(n, 3) * sigmas
    noisy_actions = targets + perturbation + noise
    corrections = targets - noisy_actions

    if use_estimated:
        k_samples = np.random.randn(n, 10, 3) * sigmas[:, None, :]
        feat_sigmas = k_samples.std(axis=1)
    else:
        feat_sigmas = sigmas.copy()

    return noisy_actions, corrections, feat_sigmas


class CorrectionMLP(nn.Module):
    def __init__(self, d_in, d_hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(),
            nn.Linear(d_hid, d_hid), nn.ReLU(),
            nn.Linear(d_hid, 3))

    def forward(self, x):
        return self.net(x)


def make_input(obs, sigmas, mode):
    if mode == 'scalar':
        return np.hstack([obs, sigmas.mean(1, keepdims=True)])
    elif mode == 'per_dof':
        return np.hstack([obs, sigmas])
    return obs


def train_eval(X_tr, y_tr, X_te, y_te, d_in, epochs=100, bs=512, lr=1e-3):
    Xtr = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)
    yte = torch.tensor(y_te, dtype=torch.float32, device=DEVICE)
    model = CorrectionMLP(d_in).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = Xtr.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xte)
        ax_mse = ((pred - yte)**2).mean(0).cpu().numpy()
    return dict(mse_x=float(ax_mse[0]), mse_y=float(ax_mse[1]),
                mse_z=float(ax_mse[2]), mse_total=float(ax_mse.mean()))


def run_group(gid, sx, sy, sz, est):
    set_seed(SEED)
    obs_tr, cor_tr, sig_tr = generate_data(20000, sx, sy, sz, est)
    obs_te, cor_te, sig_te = generate_data(5000, sx, sy, sz, est)
    tag = "K=10" if est else "oracle"
    res = {}
    for mode, lbl in [('scalar','A(scalar)'),('per_dof','B(per-DOF)'),('none','C(none)')]:
        set_seed(SEED)
        Xtr = make_input(obs_tr, sig_tr, mode)
        Xte = make_input(obs_te, sig_te, mode)
        m = train_eval(Xtr, cor_tr, Xte, cor_te, Xtr.shape[1])
        res[lbl] = m
        print(f"  G{gid} [{tag}] {lbl}: x={m['mse_x']:.6f} y={m['mse_y']:.6f} z={m['mse_z']:.6f} total={m['mse_total']:.6f}", flush=True)
    return res


def check(gid, r):
    A, B, C = r['A(scalar)'], r['B(per-DOF)'], r['C(none)']
    yi = (A['mse_y']-B['mse_y'])/A['mse_y']
    zi = (A['mse_z']-B['mse_z'])/A['mse_z']
    xok = B['mse_x'] <= A['mse_x']*1.05
    c1 = yi>0.10 and zi>0.10 and xok
    print(f"  C1(B vs A): y_imp={yi:.4f} z_imp={zi:.4f} x_ok={xok} -> {'PASS' if c1 else 'FAIL'}", flush=True)
    imps = [(C[a]-B[a])/C[a] for a in ['mse_x','mse_y','mse_z']]
    c2 = any(i>=0.05 for i in imps)
    print(f"  C2(B vs C): imps={[f'{v:.4f}' for v in imps]} -> {'PASS' if c2 else 'FAIL'}", flush=True)
    return c1 and c2


def main():
    print("="*80, flush=True)
    print("3D Point-Reaching MVP: per-DOF vs scalar vs no uncertainty", flush=True)
    print(f"Design: bias = sigma * {ALPHA}; obs = noisy_action only", flush=True)
    print("="*80, flush=True)
    cfgs = [(1,.5,.1,.1,False),(2,.2,.1,.1,False),(3,.15,.1,.1,False),
            (4,.5,.1,.1,True),(5,.2,.1,.1,True),(6,.15,.1,.1,True)]
    AR, AP = {}, {}
    for gid,sx,sy,sz,est in cfgs:
        tag = "K=10" if est else "oracle"
        print(f"\n--- Group {gid}: sigma=({sx},{sy},{sz}), unc={tag} ---", flush=True)
        AR[gid] = run_group(gid,sx,sy,sz,est)
        if gid in [1,4]:
            print(f"\n  Pass check Group {gid}:", flush=True)
            AP[gid] = check(gid, AR[gid])
            print(f"  Group {gid}: {'PASS' if AP[gid] else 'FAIL'}", flush=True)

    print("\n"+"="*80, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("="*80, flush=True)
    print(f"{'Grp':>4} {'Model':>12} {'MSE_x':>10} {'MSE_y':>10} {'MSE_z':>10} {'Total':>10}", flush=True)
    print("-"*60, flush=True)
    for g in sorted(AR):
        for ml in ['A(scalar)','B(per-DOF)','C(none)']:
            m=AR[g][ml]
            print(f"{g:>4} {ml:>12} {m['mse_x']:>10.6f} {m['mse_y']:>10.6f} {m['mse_z']:>10.6f} {m['mse_total']:>10.6f}", flush=True)
        print(flush=True)

    print("="*80, flush=True)
    print("VERDICT", flush=True)
    print("="*80, flush=True)
    ok=True
    for g in sorted(AP):
        s="PASS" if AP[g] else "FAIL"
        print(f"  Group {g}: {s}", flush=True)
        if not AP[g]: ok=False
    v="SUCCESS" if ok else "NEGATIVE"
    print(f"\n  OVERALL: {v}", flush=True)
    print("="*80, flush=True)


if __name__=="__main__":
    main()
