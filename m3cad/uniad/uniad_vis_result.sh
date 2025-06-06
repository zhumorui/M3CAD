#!/bin/bash

python ./tools/analysis_tools/visualize/run.py \
    --predroot output/results.pkl \
    --out_folder ./output/ \
    --demo_video output_video.mp4 \
    --project_to_cam True