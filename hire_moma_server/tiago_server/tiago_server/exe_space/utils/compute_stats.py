#!/usr/bin/env python3
"""Fit the action scaler / state standardizer on raw demonstration data.

  python -m tiago_server.exe_space.utils.compute_stats --data_dir data/place_only/imitation --data_dir data/place_only/warmup \
      --out stats/place_only.json
"""
from absl import app, flags

from hire_moma_launcher.wrappers.normalization import compute_stats, load_transitions, save_stats

FLAGS = flags.FLAGS
flags.DEFINE_multi_string("data_dir", None, "Folders with raw *.pkl data (imitation and/or warm-up).")
flags.DEFINE_string("out", None, "Output json.")


def main(_):
    transitions = load_transitions(FLAGS.data_dir)
    save_stats(compute_stats(transitions), FLAGS.out)
    print(f"Fitted on {len(transitions)} transitions -> {FLAGS.out}")


if __name__ == "__main__":
    app.run(main)
