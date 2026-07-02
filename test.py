'''
 * SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution 
 * Modified from diffusers by Rongyuan Wu
 * 24/12/2023
'''
import os
import sys
sys.path.append(os.getcwd())
import cv2
import glob
import argparse
import numpy as np
import yaml
from PIL import Image

import torch
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)

from wavelet_color_fix import wavelet_color_fix, adain_color_fix

from ramseesr.models.ram_lora import ram
from ramseesr import inference_ram as inference
from ramseesr import get_transform

from typing import Mapping, Any
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import os
os.environ['http_proxy'] = 'http://nbproxy.mlp.oppo.local:8888'
os.environ['https_proxy'] = 'http://nbproxy.mlp.oppo.local:8888'

logger = get_logger(__name__, log_level="INFO")


tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
def load_state_dict_diffbirSwinIR(model: nn.Module, state_dict: Mapping[str, Any], strict: bool=False) -> None:
    state_dict = state_dict.get("state_dict", state_dict)
    
    is_model_key_starts_with_module = list(model.state_dict().keys())[0].startswith("module.")
    is_state_dict_key_starts_with_module = list(state_dict.keys())[0].startswith("module.")
    
    if (
        is_model_key_starts_with_module and
        (not is_state_dict_key_starts_with_module)
    ):
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    if (
        (not is_model_key_starts_with_module) and
        is_state_dict_key_starts_with_module
    ):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=strict)


def load_pipeline(args, accelerator, enable_xformers_memory_efficient_attention):
    
    # Load scheduler, tokenizer and models.
    
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    # feature_extractor = CLIPImageProcessor.from_pretrained(f"{args.pretrained_model_path}/feature_extractor")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_path, subfolder="controlnet")
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the validation pipeline
    # validation_pipeline = StableDiffusionControlNetPipeline(
    #     vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor, 
    #     unet=unet, controlnet=controlnet, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
    # )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    validation_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    validation_pipeline.scheduler = UniPCMultistepScheduler.from_config(validation_pipeline.scheduler.config)
    validation_pipeline = validation_pipeline.to(accelerator.device)
    validation_pipeline.set_progress_bar_config(disable=True)

    return validation_pipeline

def load_tag_model(args, device='cuda'):
    
    model = ram(pretrained='/home/notebook/code/personal/S9048624/sr/SeeSR/preset/models/ram_swin_large_14m.pth',
                pretrained_condition=args.ram_ft_path,
                image_size=384,
                vit='swin_l')
    model.eval()
    model.to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"The total number of RAM parameters is: {params}")


    return model
    
def get_validation_prompt(args, image, model, device='cuda'):
    validation_prompt = ""
 
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq = ram_transforms(lq)
    res = inference(lq, model)
    ram_encoder_hidden_states = model.generate_image_embeds(lq)

    validation_prompt = f"{res[0]}, {args.prompt},"

    return validation_prompt, ram_encoder_hidden_states

def main(args, enable_xformers_memory_efficient_attention=True,):
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)
            elif not hasattr(args, key):
                setattr(args, key, value)

    txt_path = os.path.join(args.output_dir, 'txt')
    os.makedirs(txt_path, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("ControlNet")

    pipeline = load_pipeline(args, accelerator, enable_xformers_memory_efficient_attention)
    model = load_tag_model(args, accelerator.device)
 
    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)
        # print('======== loading dataset ', os.path.isdir(args.image_path))
        if os.path.isdir(args.image_path):
            image_names = sorted(glob.glob(args.image_path+'/*'))
        else:
            image_names = [args.image_path]
        # print(image_names)
        time_list = []
        for image_idx, image_name in enumerate(image_names[:]):
            print(f'================== process {image_idx} imgs... ===================')
            validation_image = Image.open(image_name).convert("RGB")
            import time
            start_time = time.time()
            validation_prompt, ram_encoder_hidden_states = get_validation_prompt(args, validation_image, model)
            validation_prompt += args.added_prompt # clean, extremely detailed, best quality, sharp, clean
            negative_prompt = args.negative_prompt #dirty, messy, low quality, frames, deformed, 
            
            if args.save_prompts:
                txt_save_path = f"{txt_path}/{os.path.basename(image_name).split('.')[0]}.txt"
                file = open(txt_save_path, "w")
                file.write(validation_prompt)
                file.close()
            print(f'{validation_prompt}')

            ori_width, ori_height = validation_image.size
            resize_flag = False
            rscale = args.upscale
            if ori_width < args.process_size//rscale or ori_height < args.process_size//rscale:
                scale = (args.process_size//rscale)/min(ori_width, ori_height)
                tmp_image = validation_image.resize((int(scale*ori_width), int(scale*ori_height)))

                validation_image = tmp_image
                resize_flag = True

            validation_image = validation_image.resize((validation_image.size[0]*rscale, validation_image.size[1]*rscale))
            validation_image = validation_image.resize((validation_image.size[0]//8*8, validation_image.size[1]//8*8))
            width, height = validation_image.size
            resize_flag = True #

            print(f'input size: {height}x{width}', image_name)

            for sample_idx in range(args.sample_times):
                os.makedirs(f'{args.output_dir}/sample{str(sample_idx).zfill(2)}/', exist_ok=True)
            
            for sample_idx in range(args.sample_times):  
                with torch.autocast("cuda"):
                    
                    image = pipeline(
                            validation_prompt, validation_image, num_inference_steps=args.num_inference_steps, generator=generator, height=height, width=width,
                            guidance_scale=args.guidance_scale, negative_prompt=negative_prompt,
                            
                        ).images[0]

                end_time = time.time()
                time_list.append(end_time-start_time)
                if args.align_method == 'nofix':
                    image = image
                else:
                    if args.align_method == 'wavelet':
                        image = wavelet_color_fix(image, validation_image)
                    elif args.align_method == 'adain':
                        image = adain_color_fix(image, validation_image)

                if resize_flag: 
                    image = image.resize((ori_width*rscale, ori_height*rscale))
                    
                name, ext = os.path.splitext(os.path.basename(image_name))
                
                image.save(f'{args.output_dir}/sample{str(sample_idx).zfill(2)}/{name}.png')
        print(f'avg time: {np.mean(time_list)}')
        print(time_list)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlnet_model_path", type=str, default='/home/notebook/code/personal/S9048624/sr/SeeSR/preset/ckp')
    parser.add_argument("--ram_ft_path", type=str, default='/home/notebook/code/personal/S9048624/sr/SeeSR/preset/DAPE.pth')
    parser.add_argument("--pretrained_model_path", type=str, default='/home/notebook/code/personal/S9048624/sr/stable-diffusion-2-base')
    parser.add_argument("--prompt", type=str, default="") # user can add self-prompt to improve the results
    parser.add_argument("--added_prompt", type=str, default="clean, high-resolution, 8k")
    parser.add_argument("--negative_prompt", type=str, default="dotted, noise, blur, lowres, smooth")
    parser.add_argument("--image_path", type=str, default='/home/notebook/code/personal/S9048624/sr/ossr_version_0425/benchmark/RealLR200/LR')
    parser.add_argument("--output_dir", type=str, default='/home/notebook/code/personal/S9048624/sr/ossr_version_0425/benchmark/RealLR200/SeeSR5.5cfg2')
    parser.add_argument("--mixed_precision", type=str, default="fp16") # no/fp16/bf16
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--blending_alpha", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=2048) # latent size, for 24G
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=2048) # image size, for 13G
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32) 
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample_times", type=int, default=1)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--save_prompts", action='store_true')
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    args = parser.parse_args()
    main(args)



