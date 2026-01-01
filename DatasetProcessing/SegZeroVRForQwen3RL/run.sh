#!/bin/bash

# Launch n processes in parallel, one per GPU
for chunk_id in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$chunk_id python sam3_for_baseline_part_boxes_on_segzero_dataset.py \
        --chunk_id $chunk_id \
        --save_dir /data/VLMGroundingProject/Datasets/SegZeroVisualReasoner/Sam3_Boxes &
done

# Wait for all background processes to complete
wait

echo "All chunks completed!"