
cd /home/notebook/code/personal/S9048624/sr/0414RLSR/diffusers0250
pip install -e .
cd examples/controlnet
pip install -r requirements.txt
pip install basicsr fairscale loralib clean-fid peft vision_aided_loss transformers==4.35.2 pyyaml

cd /home/notebook/code/personal/S9048624/sr/clip/open_clip
pip install .

cd /home/notebook/data/group/LowLevelLLM/libs/IQA-PyTorch-main
pip install -r requirements.txt
python setup.py develop


accelerate config default
cd /home/notebook/code/personal/S9048624/sr/0414RLSR/SRMethod/ControlNet


CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7," accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path="/home/notebook/data/group/LowLevelLLM/models/diffusion_models/stable-diffusion-2-1-base" \
 --output_dir="./experiment/ControlNetSR3" \
 --root_folders '/home/notebook/data/group/LowLevelLLM/DataSets/lsdir_ffhq10k' \
 --enable_xformers_memory_efficient_attention \
 --ram_ft_path 'preset/models/DAPE.pth' \
 --mixed_precision="fp16" \
 --resolution=512 \
 --learning_rate=5e-5 \
 --train_batch_size=4 \
 --gradient_accumulation_steps=1 \
 --null_text_ratio=0.5 
 --dataloader_num_workers=0 \
 --checkpointing_steps=500 

#  --validation_image "/home/notebook/code/personal/S9048624/sr/0414RLSR/benchmark/RealSRCrop128/test_SR_bicubic/Nikon_047_LR4.png","/home/notebook/code/personal/S9048624/sr/0414RLSR/benchmark/RealSRCrop128/test_SR_bicubic/Nikon_043_LR4.png"] \
#  --validation_prompt=["beard, eye, face, man, portrait, stare","flower, pink, plant, purple"] \

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7," accelerate launch train_seesr.py \
# --pretrained_model_name_or_path="preset/models/stable-diffusion-2-base" \
# --output_dir="./experience/seesr" \
# --root_folders 'preset/datasets/training_datasets' \
# --ram_ft_path 'preset/models/DAPE.pth' \
# --enable_xformers_memory_efficient_attention \
# --mixed_precision="fp16" \
# --resolution=512 \
# --learning_rate=5e-5 \
# --train_batch_size=2 \
# --gradient_accumulation_steps=2 \
# --null_text_ratio=0.5 
# --dataloader_num_workers=0 \
# --checkpointing_steps=10000 
