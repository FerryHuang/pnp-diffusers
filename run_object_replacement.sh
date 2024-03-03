#!/bin/bash

python pnp.py \
  --config_path config_pnp.yaml \
  --json_file new_bench/object_replacement.json \
  --save_dir experiments/object_replacement \
  --device 4