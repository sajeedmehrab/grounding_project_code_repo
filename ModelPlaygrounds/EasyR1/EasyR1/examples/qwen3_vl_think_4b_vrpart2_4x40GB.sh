#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3

set -x

MODEL_PATH=/data/VLMGroundingProject/PretrainedModels/Qwen3-VL-4B-Thinking # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/data/VLMGroundingProject/Datasets/InstructPart/train1800/instructpart_train_dataset_for_segzero_partfield_840 \
    data.val_files=/data/VLMGroundingProject/Datasets/InstructPart/train1800/instructpart_train_dataset_for_segzero_partfield_840 \
    data.format_prompt=/home/ksmehrab/AttentionGrounding/ModelPlaygrounds/SegZero/EvaluationScripts/Prompts/vrpart2_prompt.txt \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_vl_think_4b_vrpart2 \
    trainer.n_gpus_per_node=4 \
    trainer.save_checkpoint_path=/data/VLMGroundingProject/ModelData/EasyR1 \
    trainer.total_epochs=3