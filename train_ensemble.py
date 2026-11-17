import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = "data"
OUT_DIR = "results"

FEATURES = ["z", "n"]
STABLE = "is_stable"
TARGET = "log10_half_life_sec"

CFG = dict(
    seed=42,
    members=20,
    hidden=(128, 128, 128),
    lr=3e-4,
    weight_decay=1e-4,
    batch_size=64,
    max_epochs=1000,
    patience=40,
    lr_patience=25,
    cls_weight=1.0,
    threshold=0.32,
)

Z_MAGIC = np.array([2, 8, 20, 28, 50, 82])
N_MAGIC = np.array([2, 8, 20, 28, 50, 82, 126])


def set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.set_default_dtype(torch.float64)


def signed_dist_to_magic(values, magic):
    values = np.asarray(values, dtype=np.float64)
    nearest = magic[np.abs(values[:, None] - magic[None, :]).argmin(axis=1)]
    return values - nearest


def make_features(z, n):
    z = np.asarray(z, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    dz = signed_dist_to_magic(z, Z_MAGIC)
    dn = signed_dist_to_magic(n, N_MAGIC)
    a = z + n
    return np.stack([z, n, n - z, z + n, z % 2, n % 2,
                     dz, np.abs(dz), dn, np.abs(dn),
                     (n - z) ** 2 / a,
                     z ** 2 / np.cbrt(a)],
                    axis=1)


def load_split(path: Path):
    df = pd.read_csv(path)
    X = torch.tensor(make_features(df["z"], df["n"]))
    y = torch.tensor(df[TARGET].to_numpy(dtype=np.float64))
    s = torch.tensor(df[STABLE].to_numpy(dtype=np.float64))
    return df, X, y, s


class Standardizer:
    def __init__(self, data: torch.Tensor):
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse(self, data):
        return data * self.std + self.mean


def make_targets(y_raw, y_scaler):
    mask = ~torch.isnan(y_raw)
    y_std = torch.where(mask, (y_raw - y_scaler.mean[0]) / y_scaler.std[0], torch.zeros_like(y_raw))
    return y_std, mask.to(y_raw.dtype)


class GaussianMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=(128, 128, 128)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 3)

    def forward(self, x):
        out = self.head(self.body(x))
        mean = out[:, 0]
        var = F.softplus(out[:, 1]) + 1e-6
        logit = out[:, 2]
        return mean, var, logit


def masked_gaussian_nll(mean, var, target, mask):
    nll = 0.5 * torch.log(var) + 0.5 * (target - mean) ** 2 / var
    return (nll * mask).sum() / mask.sum().clamp_min(1.0)


def total_loss(mean, var, logit, y, mask, stable, cls_weight):
    reg = masked_gaussian_nll(mean, var, y, mask)
    cls = F.binary_cross_entropy_with_logits(logit, stable)
    return reg + cls_weight * cls


def train_member(member_seed, train, val, cfg, verbose=True):
    Xtr, ytr, mtr, str_ = train
    Xva, yva, mva, sva = val

    torch.manual_seed(member_seed)
    model = GaussianMLP(in_dim=Xtr.shape[1], hidden=cfg["hidden"])

    shuffle_gen = torch.Generator()
    shuffle_gen.manual_seed(member_seed)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=cfg["lr_patience"])

    n = Xtr.shape[0]
    best_val, best_state, best_epoch, wait = float("inf"), None, 0, 0
    history = []

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(n, generator=shuffle_gen)
        train_loss_sum = 0.0
        for start in range(0, n, cfg["batch_size"]):
            idx = perm[start:start + cfg["batch_size"]]
            mean, var, logit = model(Xtr[idx])
            loss = total_loss(mean, var, logit, ytr[idx], mtr[idx], str_[idx], cfg["cls_weight"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * len(idx)
        train_loss = train_loss_sum / n

        model.eval()
        with torch.no_grad():
            vmean, vvar, vlogit = model(Xva)
            val_loss = total_loss(vmean, vvar, vlogit, yva, mva, sva, cfg["cls_weight"]).item()
        sched.step(val_loss)
        history.append((epoch, train_loss, val_loss))

        if val_loss < best_val - 1e-6:
            best_val, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break

    model.load_state_dict(best_state)
    model.eval()
    if verbose:
        print(f"    seed {member_seed}: stopped at epoch {epoch}, best epoch {best_epoch}, "
              f"best val loss {best_val:.4f}")
    return model, history, best_epoch, best_val


@torch.no_grad()
def ensemble_predict(models, X):
    means, vars_, probs = [], [], []
    for m in models:
        m.eval()
        mu, var, logit = m(X)
        means.append(mu)
        vars_.append(var)
        probs.append(torch.sigmoid(logit))
    means = torch.stack(means)
    vars_ = torch.stack(vars_)
    probs = torch.stack(probs)
    ens_mean = means.mean(dim=0)
    ens_var = (vars_ + means ** 2).mean(dim=0) - ens_mean ** 2
    ens_var = ens_var.clamp_min(1e-12)
    return {"mean": ens_mean, "var": ens_var, "member_means": means, "member_probs": probs,
            "prob_stable": probs.mean(dim=0)}


def regression_metrics(y_true, mean, var):
    err = mean - y_true
    sigma = var.sqrt()
    z = err.abs() / sigma
    return {
        "rmse": math.sqrt((err ** 2).mean().item()),
        "mae": err.abs().mean().item(),
        "r2": 1.0 - ((err ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()).item(),
        "nll": (0.5 * math.log(2 * math.pi) + 0.5 * torch.log(var) + 0.5 * err ** 2 / var).mean().item(),
        "coverage_1sigma": (z <= 1.0).double().mean().item(),
        "coverage_2sigma": (z <= 2.0).double().mean().item(),
        "mean_sigma": sigma.mean().item(),
    }


def classification_metrics(stable_true, prob, threshold):
    pred = (prob >= threshold).to(stable_true.dtype)
    tp = ((pred == 1) & (stable_true == 1)).sum().item()
    fp = ((pred == 1) & (stable_true == 0)).sum().item()
    fn = ((pred == 0) & (stable_true == 1)).sum().item()
    tn = ((pred == 0) & (stable_true == 0)).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    pos, neg = prob[stable_true == 1], prob[stable_true == 0]
    if len(pos) > 0 and len(neg) > 0:
        diff = pos[:, None] - neg[None, :]
        auc = (((diff > 0).double().sum() + 0.5 * (diff == 0).double().sum()) / (len(pos) * len(neg))).item()
    else:
        auc = float("nan")

    eps = 1e-12
    p = prob.clamp(eps, 1 - eps)
    log_loss = -(stable_true * torch.log(p) + (1 - stable_true) * torch.log(1 - p)).mean().item()
    return {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": precision, "recall": recall, "f1": f1, "auc": auc,
        "log_loss": log_loss, "brier": ((prob - stable_true) ** 2).mean().item(),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def to_original_units(mean_std, var_std, y_scaler):
    ystd = y_scaler.std[0]
    return mean_std * ystd + y_scaler.mean[0], var_std * ystd ** 2


def main() -> None:
    cfg = dict(CFG)

    set_global_determinism(cfg["seed"])
    data_dir, out_dir = Path(DATA_DIR), Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"PyTorch {torch.__version__} | device: cpu | float64 | seed {cfg['seed']} | members {cfg['members']}")

    tr_df, Xtr_raw, ytr_raw, str_ = load_split(data_dir / "train.csv")
    va_df, Xva_raw, yva_raw, sva = load_split(data_dir / "val.csv")
    te_df, Xte_raw, yte_raw, ste = load_split(data_dir / "test.csv")
    print(f"Data: train {len(tr_df)} ({int(str_.sum())} stable), val {len(va_df)} ({int(sva.sum())} stable), "
          f"test {len(te_df)} ({int(ste.sum())} stable)")

    x_scaler = Standardizer(Xtr_raw)
    y_scaler = Standardizer(ytr_raw[~torch.isnan(ytr_raw)].unsqueeze(1))
    Xtr, Xva, Xte = (x_scaler.transform(X) for X in (Xtr_raw, Xva_raw, Xte_raw))
    ytr, mtr = make_targets(ytr_raw, y_scaler)
    yva, mva = make_targets(yva_raw, y_scaler)
    train = (Xtr, ytr, mtr, str_)
    val = (Xva, yva, mva, sva)

    print("Training members:")
    models, histories = [], []
    for i in range(cfg["members"]):
        member_seed = cfg["seed"] + 1 + i
        model, hist, _, _ = train_member(member_seed, train, val, cfg)
        models.append(model)
        histories.append(hist)

    results = {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}}

    te_has_hl = ~torch.isnan(yte_raw)
    y_te_rad = yte_raw[te_has_hl]
    train_hl = ytr_raw[~torch.isnan(ytr_raw)]
    results["baseline_half_life_test"] = regression_metrics(
        y_te_rad, torch.full_like(y_te_rad, train_hl.mean().item()), torch.full_like(y_te_rad, train_hl.var().item()))
    results["baseline_stability_test"] = classification_metrics(ste, torch.zeros_like(ste), cfg["threshold"])

    for name, X, y_raw, s in (("val", Xva, yva_raw, sva), ("test", Xte, yte_raw, ste)):
        pred = ensemble_predict(models, X)
        mean_o, var_o = to_original_units(pred["mean"], pred["var"], y_scaler)
        has_hl = ~torch.isnan(y_raw)

        results[f"half_life_{name}"] = regression_metrics(y_raw[has_hl], mean_o[has_hl], var_o[has_hl])
        results[f"stability_{name}"] = classification_metrics(s, pred["prob_stable"], cfg["threshold"])

        member_rmse = []
        for mm in pred["member_means"]:
            mm_o = mm * y_scaler.std[0] + y_scaler.mean[0]
            member_rmse.append(math.sqrt(((mm_o[has_hl] - y_raw[has_hl]) ** 2).mean().item()))
        results[f"single_member_half_life_rmse_{name}"] = member_rmse

        if name == "test":
            pred_df = te_df.copy()
            pred_df["p_stable"] = pred["prob_stable"].numpy()
            pred_df["pred_is_stable"] = (pred["prob_stable"] >= cfg["threshold"]).numpy().astype(int)
            pred_df["pred_log10_half_life"] = mean_o.numpy()
            pred_df["pred_sigma"] = var_o.sqrt().numpy()
            pred_df["half_life_error"] = pred_df["pred_log10_half_life"] - pred_df[TARGET]
            pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

            n_m, n_s = pred["member_means"].shape
            member_df = pd.DataFrame({
                "z": np.tile(te_df["z"].to_numpy(), n_m),
                "n": np.tile(te_df["n"].to_numpy(), n_m),
                "member": np.repeat(np.arange(n_m), n_s),
                "pred_log10_half_life": (pred["member_means"] * y_scaler.std[0] + y_scaler.mean[0]).numpy().ravel(),
                "p_stable": pred["member_probs"].numpy().ravel(),
            })
            member_df.to_csv(out_dir / "test_member_predictions.csv", index=False)

    torch.save(
        {
            "member_state_dicts": [m.state_dict() for m in models],
            "x_mean": x_scaler.mean, "x_std": x_scaler.std,
            "y_mean": y_scaler.mean, "y_std": y_scaler.std,
            "config": cfg,
        },
        out_dir / "ensemble.pt",
    )
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    def reg_row(label, m):
        print(f"  {label:30s} RMSE {m['rmse']:.3f} | MAE {m['mae']:.3f} | R2 {m['r2']:.3f} | "
              f"NLL {m['nll']:.3f} | 1-sigma cov {m['coverage_1sigma']:.2f} | 2-sigma cov {m['coverage_2sigma']:.2f}")

    def cls_row(label, m):
        print(f"  {label:30s} acc {m['accuracy']:.3f} | precision {m['precision']:.3f} | recall {m['recall']:.3f} | "
              f"F1 {m['f1']:.3f} | AUC {m['auc']:.3f} | TP {m['tp']} FP {m['fp']} FN {m['fn']} TN {m['tn']}")

    print("\nHALF-LIFE (radioactive nuclides; log10 seconds; RMSE 1.0 = typically off by a factor of 10)")
    reg_row("Baseline (train mean), test", results["baseline_half_life_test"])
    reg_row("Ensemble, validation", results["half_life_val"])
    reg_row("Ensemble, TEST", results["half_life_test"])
    print("  Single-member test RMSEs:", ", ".join(f"{r:.3f}" for r in results["single_member_half_life_rmse_test"]))

    print("\nSTABILITY (positive class = stable)")
    cls_row("Baseline (all radioactive), test", results["baseline_stability_test"])
    cls_row("Ensemble, validation", results["stability_val"])
    cls_row("Ensemble, TEST", results["stability_test"])
    print(f"\nSaved to {out_dir}/: metrics.json, test_predictions.csv, ensemble.pt")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, h in enumerate(histories):
        ax.plot([r[0] for r in h], [r[2] for r in h], label=f"member {i}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss (combined)")
    ax.set_title("Validation loss per member")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, ncol=2, fontsize=8)
    path = out_dir / "validation_loss.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")

    rad = pred_df[pred_df[STABLE] == 0]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.errorbar(rad[TARGET], rad["pred_log10_half_life"], yerr=rad["pred_sigma"],
                fmt="o", ms=2, alpha=0.4, elinewidth=0.5)
    lo, hi = rad[TARGET].min(), rad[TARGET].max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("true log10 half-life (s)")
    ax.set_ylabel("predicted (±1 sigma)")
    ax.set_title("Test set (radioactive): predicted vs true")
    path = out_dir / "predicted_vs_true.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(pred_df["n"], pred_df["z"], c=pred_df["p_stable"], cmap="viridis", s=8, vmin=0, vmax=1)
    ax.set_xlabel("N")
    ax.set_ylabel("Z")
    ax.set_title("Test set: predicted p(stable)")
    fig.colorbar(sc, ax=ax)
    path = out_dir / "predicted_p_stable.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")


if __name__ == "__main__":
    main()