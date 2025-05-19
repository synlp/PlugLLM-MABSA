# PlugLLM-MABSA


This is the code of our paper.

## Train

You need to change the parameters `path_to_llava`, `path_to_train_file`, `path_to_image_folder`, 
`path_to_clip`, `path_to_output_file`, `path_to_vit_base`, and `path_to_bert`.

```bash
deepspeed --include localhost --master_port=12345 train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path path_to_llava \
    --version v1 \
    --data_path path_to_train_file \
    --image_folder path_to_image_folder \
    --vision_tower path_to_clip \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir path_to_output_file \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --visual_plugin \
    --visual_plugin_model_path path_to_vit_base \
    --text_plugin \
    --text_plugin_model_path path_to_bert \
    --gcn_layer_num 3 \
    --use_hub \
    --hub_memory_size 20 \
    --hub_output_size 8 \
    --hub_hidden_size 768
```
