#!/bin/bash

# # Launch n processes in parallel, one per GPU
# for chunk_id in 0 1 2; do
#     CUDA_VISIBLE_DEVICES=$((chunk_id+1)) python sam3_on_pascalpart.py \
#         --chunk_id $chunk_id \
#         --save_dir /data/VLMGroundingProject/BaselineResults/PascalPart/SAM3 \
#         --total_chunks 3 \
#         --batch_size 32 &
# done

# # Wait for all background processes to complete
# wait

# echo "All chunks completed!"


# Launch a single process (for testing or small runs)
# CUDA_VISIBLE_DEVICES=0 python sam3_for_baseline_part_boxes_on_instructpart.py \
#     --chunk_id 0 \
#     --save_dir /data/VLMGroundingProject/Datasets/InstructPart/train1800 \
#     --total_chunks 2 \
#     --batch_size 32

# CUDA_VISIBLE_DEVICES=2 python sam3_for_baseline_part_boxes_on_instructpart.py \
#     --chunk_id 1 \
#     --save_dir /data/VLMGroundingProject/Datasets/InstructPart/train1800 \
#     --total_chunks 2 \
#     --batch_size 32