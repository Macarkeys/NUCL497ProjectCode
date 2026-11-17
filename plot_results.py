from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgb
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

DATA_DIR = "data"
RESULTS_DIR = "results"
UNKNOWN_CSV = "nuclides_unknown_half_life.csv"
CMAP = "viridis_r"

TARGET = "log10_half_life_sec"
TARGET_UNC = "log10_half_life_unc"

COLORS = {
    "background": "#ffffff",
    "unused": "#494949",
    "train": "#000000",
    "val": "#000000",
    "stable_ok": "#009e47",
    "radio_ok": "#98f0b4",
    "stable_bad": "#a8001a",
    "radio_bad": "#ff6b5e",
    "within_csv": "#FF3300",
}


def classify_stability(test_df: pd.DataFrame) -> pd.Series:
    stable = test_df["is_stable"].to_numpy() == 1
    pred_stable = test_df["pred_is_stable"].to_numpy() == 1
    keys = np.select([stable & pred_stable, ~stable & ~pred_stable, stable & ~pred_stable, ~stable & pred_stable],
                     ["stable_ok", "radio_ok", "stable_bad", "radio_bad"], default="radio_ok")
    return pd.Series(keys, index=test_df.index)


def sigma_distance(rad: pd.DataFrame):
    err = (rad["pred_log10_half_life"] - rad[TARGET]).abs().to_numpy()
    k = err / rad["pred_sigma"].to_numpy()
    csv_unc = rad[TARGET_UNC].to_numpy()
    in_csv = np.isfinite(csv_unc) & (err <= csv_unc)
    return k, in_csv


def base_image(train, val, unknown, z_max, n_max) -> np.ndarray:
    img = np.ones((z_max + 1, n_max + 1, 3))
    img[:, :] = to_rgb(COLORS["background"])
    if len(unknown):
        img[unknown.z.to_numpy(int), unknown.n.to_numpy(int)] = to_rgb(COLORS["unused"])
    img[train.z.to_numpy(int), train.n.to_numpy(int)] = to_rgb(COLORS["train"])
    img[val.z.to_numpy(int), val.n.to_numpy(int)] = to_rgb(COLORS["val"])
    return img


def draw_chart(img, handles, title, out_path, z_max, n_max, cbar=None):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(img, origin="lower", interpolation="nearest",
              extent=(-0.5, n_max + 0.5, -0.5, z_max + 0.5), aspect="equal")
    ax.set_xlabel("Neutron number N")
    ax.set_ylabel("Proton number Z")
    ax.set_title(title)
    ax.legend(handles=handles, loc="upper left", framealpha=0.95)
    if cbar is not None:
        cb = fig.colorbar(ScalarMappable(norm=cbar["norm"], cmap=cbar["cmap"]), ax=ax, fraction=0.03, pad=0.01,
                          ticks=cbar["ticks"])
        cb.ax.set_yticklabels(cbar["ticklabels"])
        cb.set_label(cbar["label"])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


def main() -> None:
    data_dir, res_dir = Path(DATA_DIR), Path(RESULTS_DIR)
    out_stab = res_dir / "chart_of_nuclides_stability.png"
    out_hl = res_dir / "chart_of_nuclides_halflife.png"

    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(res_dir / "test_predictions.csv")

    unknown_path = Path(UNKNOWN_CSV)
    if unknown_path.exists():
        unknown = pd.read_csv(unknown_path, usecols=lambda c: c in ("z", "n", "unknown_reason"))
        used_df = pd.concat([train, val, test])
        overlap = set(zip(used_df.z, used_df.n)) & set(zip(unknown.z, unknown.n))
        if overlap:
            print(f"WARNING: {len(overlap)} nuclides appear both in the unknown CSV and in train/val/test; "
                  "they will be drawn with their train/val/test colour.")
    else:
        print(f"WARNING: {unknown_path} not found; no light grey boxes will be drawn.")
        unknown = pd.DataFrame({"z": [], "n": []}).astype(int)

    z_max = int(max(train.z.max(), val.z.max(), test.z.max(), unknown.z.max() if len(unknown) else 0))
    n_max = int(max(train.n.max(), val.n.max(), test.n.max(), unknown.n.max() if len(unknown) else 0))

    base = base_image(train, val, unknown, z_max, n_max)
    n_unused = int((base == np.array(to_rgb(COLORS["unused"]))).all(axis=2).sum())
    grey_handles = [
        Patch(color=COLORS["unused"], label=f"Unused ({n_unused})"),
        Patch(color=COLORS["train"], label=f"Train ({len(train)})"),
        Patch(color=COLORS["val"], label=f"Validation ({len(val)})"),
    ]

    test = test.copy()
    test["stab"] = classify_stability(test)
    img = base.copy()
    for key in ("radio_ok", "stable_ok", "radio_bad", "stable_bad"):
        sub = test[test.stab == key]
        img[sub.z.to_numpy(int), sub.n.to_numpy(int)] = to_rgb(COLORS[key])
    sc = test.stab.value_counts()
    handles = grey_handles + [
        Patch(color=COLORS["stable_ok"], label=f"Test, correct: stable predicted stable ({sc.get('stable_ok', 0)})"),
        Patch(color=COLORS["radio_ok"], label=f"Test, correct: radioactive predicted radioactive ({sc.get('radio_ok', 0)})"),
        Patch(color=COLORS["stable_bad"], label=f"Test, wrong: stable predicted radioactive ({sc.get('stable_bad', 0)})"),
        Patch(color=COLORS["radio_bad"], label=f"Test, wrong: radioactive predicted stable ({sc.get('radio_bad', 0)})"),
    ]
    draw_chart(img, handles, "Chart of the nuclides: stability prediction on the test set", out_stab, z_max, n_max)

    rad = test[test.is_stable == 0].copy()
    k, in_csv = sigma_distance(rad)
    rest = ~in_csv
    k_max = float(k[rest].max())
    cmap = plt.get_cmap(CMAP)
    norm = Normalize(vmin=0.0, vmax=k_max)

    colors = cmap(norm(k))[:, :3]
    colors[in_csv] = to_rgb(COLORS["within_csv"])

    img = base.copy()
    stable_test = test[test.is_stable == 1]
    img[stable_test.z.to_numpy(int), stable_test.n.to_numpy(int)] = to_rgb(COLORS["background"])
    img[rad.z.to_numpy(int), rad.n.to_numpy(int)] = colors

    ticks = [t for t in (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20) if t < 0.92 * k_max] + [k_max]
    ticklabels = [f"{t:g}\u03c3" for t in ticks[:-1]] + [f"{k_max:.1f}\u03c3 (farthest)"]
    cbar = dict(norm=norm, cmap=cmap, ticks=ticks, ticklabels=ticklabels,
                label="Distance of the prediction from the true half-life (ensemble \u03c3), linear scale")
    handles = grey_handles + [
        Patch(color=COLORS["within_csv"], label=f"Test: within CSV uncertainty ({int(in_csv.sum())})"),
        Patch(facecolor="white", edgecolor="black", label=f"Test: stable, no half-life ({len(stable_test)})"),
    ]
    draw_chart(img, handles, "Chart of the nuclides: half-life error of the radioactive test nuclides",
               out_hl, z_max, n_max, cbar=cbar)

    print(f"Test nuclides: {len(test)} ({len(stable_test)} stable, {len(rad)} radioactive)")
    print("Stability prediction:")
    for key, label in (("stable_ok", "stable -> stable (correct)"), ("radio_ok", "radioactive -> radioactive (correct)"),
                       ("stable_bad", "stable -> radioactive (wrong)"), ("radio_bad", "radioactive -> stable (wrong)")):
        print(f"  {sc.get(key, 0):4d}  {label}")
    print("Half-life error of radioactive test nuclides (the chart itself uses a continuous scale):")
    print(f"  {int(in_csv.sum()):4d} ({100 * in_csv.mean():5.1f}%)  within CSV uncertainty")
    edges = [(-np.inf, 1, "within 1\u03c3"), (1, 2, "1-2\u03c3"), (2, 3, "2-3\u03c3"), (3, 4, "3-4\u03c3")]
    for lo, hi, label in edges:
        c = int((rest & (k > lo) & (k <= hi)).sum())
        print(f"  {c:4d} ({100 * c / len(rad):5.1f}%)  {label} (not within CSV)")
    c = int((rest & (k > 4)).sum())
    print(f"  {c:4d} ({100 * c / len(rad):5.1f}%)  beyond 4\u03c3 (farthest: {k_max:.1f}\u03c3)")
    if len(unknown) and "unknown_reason" in unknown.columns:
        print("Unused nuclides (light grey) by reason:")
        for reason, c in unknown.unknown_reason.value_counts().items():
            print(f"  {c:4d}  {reason}")

    plt.show()


if __name__ == "__main__":
    main()