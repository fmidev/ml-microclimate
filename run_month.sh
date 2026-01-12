#!/bin/bash
# run_month.sh

# Set a lower limit than Slurm's per-node mem
ulimit -v $((220 * 1024 * 1024))  # 220 GiB

python generate_spatiotemporal_data.py "$@"

