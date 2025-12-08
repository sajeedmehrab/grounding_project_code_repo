# VR and Qwen2.5 models:
# for chunk_id in 0 1 2 3; do
#   CUDA_VISIBLE_DEVICES=$chunk_id python infer_vrpart_on_instructpart.py \
#    --reasoning_model_path /data/VLMGroundingProject/PretrainedModels/Qwen3-VL-8B-Thinking \
#    --save_dir /data/VLMGroundingProject/BaselineResults/InstructPart/Qwen3VL_8B_Test \
#    --prompt_type vrpart2 \
#    --batch_size 32 \
#    --chunk_id $chunk_id &
# done

# Qwen3 models:
for chunk_id in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$chunk_id python infer_qwen3vl_on_instructpart.py \
   --reasoning_model_path /data/VLMGroundingProject/PretrainedModels/Qwen3-VL-8B-Thinking \
   --save_dir /data/VLMGroundingProject/BaselineResults/InstructPart/Qwen3VL_8B_Test \
   --prompt_type vrpart2 \
   --batch_size 32 \
   --chunk_id $chunk_id &
done


# Wait for all background processes to finish
wait

echo "All inference processes completed"