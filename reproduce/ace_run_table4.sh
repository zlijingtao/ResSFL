#!/bin/bash
cd "$(dirname "$0")"
bash table4/ace_run_comparison_baseline_addnoise_laplacian_2c.sh
# The first one must be ran first, feel free to alternate scripts below
bash table4/ace_run_comparison_baseline_addnoise_dropout_2c.sh
bash table4/ace_run_comparison_baseline_addnoise_topkprune_2c.sh
bash table4/ace_run_comparison_baseline_addnoise_advnoise_2c.sh
bash table4/ace_run_comparison_bottleneck_2c.sh
bash table4/ace_run_comparison_nopeek_2c.sh