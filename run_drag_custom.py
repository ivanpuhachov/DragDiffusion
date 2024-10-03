import pickle
import os
import cv2
import numpy as np
import gradio as gr
from copy import deepcopy
from einops import rearrange
from types import SimpleNamespace
from torchvision.utils import save_image
from pytorch_lightning import seed_everything

import datetime
import PIL
from PIL import Image
from PIL.ImageOps import exif_transpose
import torch
import torch.nn.functional as F

from diffusers import DDIMScheduler, AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.models.embeddings import ImageProjection
from drag_pipeline import DragPipeline


from utils.ui_utils import preprocess_image
from utils.drag_utils import drag_diffusion_update
from utils.attn_utils import register_attention_editor_diffusers, MutualSelfAttentionControl

def run_drag(
        source_image,  # 512,512,3 numpy image [0,255]
        image_with_clicks,  # 512 512 3, used only to store results
        mask,  # 512, 512 numpy mask, 255 is inside
        prompt,
        points,  # list of pixel coordinates of [[handle], [target], [h2],[t2]] points
        inversion_strength=0.7,  # (0,1) - at which level optimization happens
        lam=0.1,  # opt parameter
        latent_lr=0.01,
        n_pix_step=80,  # n optimization steps
        model_path="runwayml/stable-diffusion-v1-5",  # "runwayml/stable-diffusion-v1-5"
        vae_path="default",  # "default"
        lora_path="",  # "./lora_tmp"
        start_step=0,  # used in MutualSelfAttentionControl only -> the step to start mutual self-attention control
        start_layer=10,  # used in MutualSelfAttentionControl only -> the layer to start mutual self-attention control
        save_dir="./results",
        save_seq=True,
    ):
    # # Save inputs using pickle
    # inputs = {
    #     'source_image': source_image,
    #     'image_with_clicks': image_with_clicks,
    #     'mask': mask,
    #     'prompt': prompt,
    #     'points': points,
    #     'inversion_strength': inversion_strength,
    #     'lam': lam,
    #     'latent_lr': latent_lr,
    #     'n_pix_step': n_pix_step,
    #     'model_path': model_path,
    #     'vae_path': vae_path,
    #     'lora_path': lora_path,
    #     'start_step': start_step,
    #     'start_layer': start_layer
    # }
    
    # with open('inputs.pkl', 'wb') as f:
    #     pickle.dump(inputs, f)

    # initialize model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False, steps_offset=1)
    model: DragPipeline = DragPipeline.from_pretrained(model_path, scheduler=scheduler, torch_dtype=torch.float16)
    
    # call this function to override unet forward function,
    # so that intermediate features are returned after forward
    # The only difference from diffusers:
    # return intermediate UNet features of all UpSample blocks
    model.modify_unet_forward()

    # set vae
    if vae_path != "default":
        model.vae = AutoencoderKL.from_pretrained(
            vae_path
        ).to(model.vae.device, model.vae.dtype)

    # off load model to cpu, which save some memory.
    model.enable_model_cpu_offload()

    # initialize parameters
    seed = 42 # random seed used by a lot of people for unknown reason
    seed_everything(seed)

    args = SimpleNamespace()
    args.prompt = prompt
    args.points = points
    args.n_inference_step = 50  # n denoising steps from pure noise
    # we do deformations at some level (inversion_strength in [0,1]), denoise from there
    args.n_actual_inference_step = round(inversion_strength * args.n_inference_step)
    # CFG guidance
    args.guidance_scale = 1.0

    args.unet_feature_idx = [3]  # only use the output of the last UNet block

    args.r_m = 1
    args.r_p = 3
    args.lam = lam

    args.lr = latent_lr
    args.n_pix_step = n_pix_step

    full_h, full_w = source_image.shape[:2]
    args.sup_res_h = int(0.5*full_h)
    args.sup_res_w = int(0.5*full_w)

    print(args)

    source_image = preprocess_image(source_image, device, dtype=torch.float16)
    image_with_clicks = preprocess_image(image_with_clicks, device)

    # preparing editing meta data (handle, target, mask)
    mask = torch.from_numpy(mask).float() / 255.
    mask[mask > 0.0] = 1.0
    mask_original = rearrange(mask, "h w -> 1 1 h w").cuda()
    # mask is now binary, 11HW
    mask = F.interpolate(mask_original, (args.sup_res_h, args.sup_res_w), mode="nearest")  # mask has size 1 x 1 x sup_res_h x sup_res_h


    # parsing points into handle and target
    handle_points = []
    target_points = []
    # here, the point is in x,y coordinate
    for idx, point in enumerate(points):
        cur_point = torch.tensor(
            [
                point[1] / full_h * args.sup_res_h, 
                point[0] / full_w * args.sup_res_w, 
            ])
        cur_point = torch.round(cur_point)
        
        if idx % 2 == 0:
            handle_points.append(cur_point)
        else:
            target_points.append(cur_point)
    print('handle points:', handle_points)
    print('target points:', target_points)

    # set lora
    if lora_path == "":
        print("applying default parameters")
        model.unet.set_default_attn_processor()
    else:
        print("applying lora: " + lora_path)
        model.unet.load_attn_procs(lora_path)

    # obtain text embeddings
    text_embeddings = model.get_text_embeddings(prompt)

    # invert the source image
    # the latent code resolution is too small, only 64*64
    # DDIM inversion
    invert_code = model.invert(
        source_image,
        prompt,
        encoder_hidden_states=text_embeddings,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.n_inference_step,
        num_actual_inference_steps=args.n_actual_inference_step,
        )

    # empty cache to save memory
    torch.cuda.empty_cache()

    init_code = invert_code
    init_code_orig = deepcopy(init_code)
    model.scheduler.set_timesteps(args.n_inference_step)
    t = model.scheduler.timesteps[args.n_inference_step - args.n_actual_inference_step]

    # feature shape: [1280,16,16], [1280,32,32], [640,64,64], [320,64,64]
    # convert dtype to float for optimization
    init_code = init_code.float()
    text_embeddings = text_embeddings.float()
    model.unet = model.unet.float()

    updated_init_code, opt_seq = drag_diffusion_update(
        model,
        init_code,
        text_embeddings,
        t,
        handle_points,
        target_points,
        mask,
        args)
    
    print('------------')
    print('opt_seq: ', len(opt_seq))

    updated_init_code = updated_init_code.half()
    text_embeddings = text_embeddings.half()
    model.unet = model.unet.half()

    # empty cache to save memory
    torch.cuda.empty_cache()

    # hijack the attention module
    # inject the reference branch to guide the generation
    editor = MutualSelfAttentionControl(start_step=start_step,
                                        start_layer=start_layer,
                                        total_steps=args.n_inference_step,
                                        guidance_scale=args.guidance_scale)
    if lora_path == "":
        register_attention_editor_diffusers(model, editor, attn_processor='attn_proc')
    else:
        register_attention_editor_diffusers(model, editor, attn_processor='lora_attn_proc')

    # inference the synthesized image
    gen_image = model(
        prompt=args.prompt,
        encoder_hidden_states=torch.cat([text_embeddings]*2, dim=0),
        batch_size=2,
        latents=torch.cat([init_code_orig, updated_init_code], dim=0),
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.n_inference_step,
        num_actual_inference_steps=args.n_actual_inference_step
        )[1].unsqueeze(dim=0)

    # resize gen_image into the size of source_image
    # we do this because shape of gen_image will be rounded to multipliers of 8
    gen_image = F.interpolate(gen_image, (full_h, full_w), mode='bilinear')

    # save the original image, user editing instructions, synthesized image
    save_result = torch.cat([
        source_image.float() * 0.5 + 0.5,
        torch.ones((1,3,full_h,25)).cuda(),
        image_with_clicks.float() * 0.5 + 0.5,
        torch.ones((1,3,full_h,25)).cuda(),
        gen_image[0:1].float()
    ], dim=-1)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    save_image(save_result, os.path.join(save_dir, save_prefix + '.png'))
    torch.save(mask_original, os.path.join(save_dir, save_prefix +'_mask.pt'))

    if save_seq:
        os.mkdir(os.path.join(save_dir, save_prefix))
        # save list of latents in pt file
        torch.save(opt_seq, os.path.join(save_dir, save_prefix, 'opt_seq.pt'))
        for i in range(0, len(opt_seq), 2):
            # denoise latents and save
            latents = torch.cat([opt_seq[i].half(), opt_seq[i+1].half() if i+1 < len(opt_seq) else opt_seq[i].half()], dim=0)
            gen_image_seq = model(
                args.prompt,
                encoder_hidden_states=torch.cat([text_embeddings]*2, dim=0),
                batch_size=2,
                latents=latents,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.n_inference_step,
                num_actual_inference_steps=args.n_actual_inference_step
            )
            gen_image_seq = F.interpolate(gen_image_seq, (full_h, full_w), mode='bilinear')
            save_image(gen_image_seq[0].unsqueeze(dim=0), os.path.join(save_dir, save_prefix, f'iter_{i}.png'))
            if i+1 < len(opt_seq):
                save_image(gen_image_seq[1].unsqueeze(dim=0), os.path.join(save_dir, save_prefix, f'iter_{i+1}.png'))

    out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    out_image = (out_image * 255).astype(np.uint8)
    return out_image

if __name__ == "__main__":
    with open('inputs.pkl', 'rb') as f:
        inputs = pickle.load(f)
    run_drag(
        **inputs,
    )

    
    
    
