python test.py \
--pretrained_model_path /home/notebook/code/personal/S9048624/sr/stable-diffusion-2-base \
--prompt '' \
--controlnet_model_path /home/notebook/code/personal/S9048624/sr/0414RLSR/SRMethod/ControlNet/experiment/ControlNetSR/checkpoint-216000 \
--image_path /home/notebook/code/personal/S9048624/sr/0414RLSR/benchmark/RealSRCrop128/test_LR \
--output_dir experiment/ControlNetSR/results-216000-3.5 \
--num_inference_steps 20 \
--guidance_scale 3.5 \
--upscale 4
