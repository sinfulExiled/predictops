#!/usr/bin/env python
"""Generate the synthetic plant dataset.

    python generate_data.py --machines 60 --days 30 --seed 42
"""
from __future__ import annotations

import argparse
import json

from predictops.config import DATA_DIR, GeneratorConfig
from predictops.data.generator import PlantGenerator, write_dataset


def main() -> None:
    p = argparse.ArgumentParser(description="Generate PredictOps telemetry")
    p.add_argument("--machines", type=int, default=80)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=str(DATA_DIR))
    args = p.parse_args()

    cfg = GeneratorConfig(n_machines=args.machines, days=args.days,
                          seed=args.seed)
    print(f"generating {args.machines} machines x {args.days} days "
          f"(seed={args.seed}) ...")
    ds = PlantGenerator(cfg).generate()
    from pathlib import Path
    manifest = write_dataset(ds, Path(args.out))
    print(json.dumps({k: v for k, v in manifest.items() if k != "config"},
                     indent=2))
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
