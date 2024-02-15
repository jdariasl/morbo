#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
The main script for running a single replication.
"""
import os
from morbo.run_one_replication import run_one_replication
import json
import torch
import errno
from typing import Any, Dict


def fetch_data(kwargs: Dict[str, Any]) -> None:
    # this modifies kwargs in place
    problem_kwargs = kwargs.get("problem_kwargs", {})
    key = problem_kwargs.get("datapath")

    if key is not None:
        data = torch.load(key)
        problem_kwargs["data"] = data
        kwargs["problem_kwargs"] = problem_kwargs


if __name__ == "__main__":
    args = []
    args.append("main.py")
    args.append("experiments/pena")
    args.append("morbo")
    args.append(0)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(current_dir, args[1])
    config_path = os.path.join(exp_dir, "config.json")
    label = args[2]
    seed = int(float(args[3]))
    last_arg = args[4] if len(args) > 4 else None
    output_path = os.path.join(exp_dir, label, f"{str(seed).zfill(4)}_{label}.pt")

    if not os.path.exists(os.path.dirname(output_path)):
        try:
            os.makedirs(os.path.dirname(output_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(config_path, "r") as f:
        kwargs = json.load(f)
    save_callback = lambda data: torch.save(data, output_path)
    fetch_data(kwargs=kwargs)
    run_one_replication(
        seed=seed,
        label=label,
        save_callback=save_callback,
        **kwargs,
    )
