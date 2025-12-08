for chunk_id in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$chunk_id python infer_vrpart_on_partimagenet.py \
   --reasoning_model_path /data/VLMGroundingProject/ModelData/SegZero/visionreasoner_workdir/ip_vrpretrained_partreward2_nocontainment/global_step_224/actor/huggingface \
   --save_dir /data/VLMGroundingProject/BaselineResults/PartImageNet/VRPart2_NoContainment \
   --prompt_type vrpart2 \
   --batch_size 32 \
   --chunk_id $chunk_id &
done

# Wait for all background processes to finish
wait

echo "All inference processes completed"