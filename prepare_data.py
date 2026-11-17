import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_FILE = "nuclides_known_half_life.csv"
OUT_DIR = "data"
SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
FEATURES = ["z", "n"]
STABLE = "is_stable"
TARGET = "log10_half_life_sec"
TARGET_UNC = "log10_half_life_unc"


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    in_path = Path(INPUT_FILE)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, usecols=FEATURES + [STABLE, TARGET, TARGET_UNC])
    print(f"Loaded {len(df)} rows from {in_path}")

    df[STABLE] = df[STABLE].astype(np.int64)
    has_hl = np.isfinite(df[TARGET])

    n_before = len(df)
    df = df[(df[STABLE] == 1) | has_hl].copy()
    has_hl = np.isfinite(df[TARGET])
    print(f"Dropped {n_before - len(df)} rows that are neither stable nor have a half-life")
    assert not (has_hl & (df[STABLE] == 1)).any(), "A stable nuclide has a half-life value"
    assert (has_hl | (df[STABLE] == 1)).all()

    assert not df.duplicated(subset=FEATURES).any(), "Duplicate (z, n) rows found"

    df = df.sort_values(FEATURES).reset_index(drop=True)
    df[FEATURES] = df[FEATURES].astype(np.int64)

    rng = np.random.default_rng(SEED)
    train_parts, val_parts, test_parts = [], [], []
    for cls in (0, 1):
        idx = np.flatnonzero(df[STABLE].to_numpy() == cls)
        perm = rng.permutation(idx)
        n_cls = len(idx)
        n_test = int(round(TEST_FRACTION * n_cls))
        n_val = int(round(VAL_FRACTION * n_cls))
        n_train = n_cls - n_val - n_test
        train_parts.append(perm[:n_train])
        val_parts.append(perm[n_train:n_train + n_val])
        test_parts.append(perm[n_train + n_val:])

    splits = {
        "train": df.iloc[np.sort(np.concatenate(train_parts))],
        "val": df.iloc[np.sort(np.concatenate(val_parts))],
        "test": df.iloc[np.sort(np.concatenate(test_parts))],
    }

    keys = {name: set(map(tuple, s[FEATURES].to_numpy())) for name, s in splits.items()}
    assert not (keys["train"] & keys["val"]) and not (keys["train"] & keys["test"]) and not (keys["val"] & keys["test"])
    assert sum(len(s) for s in splits.values()) == len(df)

    info = {
        "seed": SEED,
        "input_file": str(in_path),
        "input_sha256": sha256_of_file(in_path),
        "features": FEATURES,
        "labels": [STABLE, TARGET],
        "n_total": int(len(df)),
    }
    for name, part in splits.items():
        part[FEATURES + [STABLE, TARGET, TARGET_UNC]].to_csv(out_dir / f"{name}.csv", index=False)
        n_stable = int(part[STABLE].sum())
        info[f"n_{name}"] = int(len(part))
        info[f"n_{name}_stable"] = n_stable
        print(f"  {name:5s}: {len(part):5d} rows ({n_stable} stable, {len(part) - n_stable} radioactive) "
              f"-> {out_dir / (name + '.csv')}")

    with open(out_dir / "split_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"Wrote {out_dir / 'split_info.json'}")


if __name__ == "__main__":
    main()