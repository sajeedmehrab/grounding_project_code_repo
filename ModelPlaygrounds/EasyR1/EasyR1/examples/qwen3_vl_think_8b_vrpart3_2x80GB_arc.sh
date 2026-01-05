#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1

set -x

MODEL_PATH=/projects/ml4science/kazi/VLMGroundingProject/PretrainedModels/Qwen3-VL-8B-Thinking # replace it with your local file path

python3 -m verl.trainer.main \
    config=/home/ksmehrab/GroundingProjectCodeRepo/ModelPlaygrounds/EasyR1/EasyR1/examples/config.yaml \
    data.train_files=/projects/ml4science/kazi/VLMGroundingProject/Datasets/SegZeroVR_InstructPart_Merged \
    data.val_files=/projects/ml4science/kazi/VLMGroundingProject/Datasets/SegZeroVR_InstructPart_Merged \
    data.format_prompt=/home/ksmehrab/GroundingProjectCodeRepo/ModelPlaygrounds/SegZero/EvaluationScripts/Prompts/vrpart2_prompt.txt \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3vlthink8b_vrpart3_try1 \
    trainer.n_gpus_per_node=2 \
    trainer.save_checkpoint_path=/projects/ml4science/kazi/VLMGroundingProject/ModelData/EasyR1/qwen3vlthink8b_vrpart3_try1 \
    trainer.total_epochs=2 \
    trainer.save_freq=556 \
    worker.reward.reward_function=/home/ksmehrab/GroundingProjectCodeRepo/ModelPlaygrounds/EasyR1/EasyR1/examples/reward_function/vision_reasoner_part3_easyr1.py:compute_score