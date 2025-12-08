for chunk_id in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$chunk_id python infer_vrpart_custom_on_pascalpart.py \
   --reasoning_model_path /data/VLMGroundingProject/ModelData/SegZero/visionreasoner_workdir/ip_vrpretrained_partreward2/global_step_224/actor/huggingface \
   --save_dir /data/VLMGroundingProject/BaselineResults/PascalPart/VRPart2 \
   --prompt_type vrpart2 \
   --chunk_id $chunk_id &
done

# Wait for all background processes to finish
wait

echo "All inference processes completed"