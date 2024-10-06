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
import copy

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
from utils.drag_utils import point_tracking, check_handle_reach_target
from utils.attn_utils import register_attention_editor_diffusers, MutualSelfAttentionControl

from my_data_loader import draw_on_image, load_points_from_theater_json, load_and_draw, get_concat_h, overlap_mask


def interpolate_feature_patch(
        feat,
        y1,
        y2,
        x1,
        x2,
):
    x1_floor = torch.floor(x1).long()
    x1_cell = x1_floor + 1
    dx = torch.floor(x2).long() - torch.floor(x1).long()

    y1_floor = torch.floor(y1).long()
    y1_cell = y1_floor + 1
    dy = torch.floor(y2).long() - torch.floor(y1).long()

    wa = (x1_cell.float() - x1) * (y1_cell.float() - y1)
    wb = (x1_cell.float() - x1) * (y1 - y1_floor.float())
    wc = (x1 - x1_floor.float()) * (y1_cell.float() - y1)
    wd = (x1 - x1_floor.float()) * (y1 - y1_floor.float())

    Ia = feat[:, :, y1_floor: y1_floor+dy, x1_floor: x1_floor+dx]
    Ib = feat[:, :, y1_cell: y1_cell+dy, x1_floor: x1_floor+dx]
    Ic = feat[:, :, y1_floor: y1_floor+dy, x1_cell: x1_cell+dx]
    Id = feat[:, :, y1_cell: y1_cell+dy, x1_cell: x1_cell+dx]

    return Ia * wa + Ib * wb + Ic * wc + Id * wd


def drag_diffusion_update(
        model: DragPipeline,
        init_code,
        text_embeddings,
        t,
        handle_points,
        target_points,
        mask,
        args,
):
    """
    Optimize init_code by moving handle_points to target_points.
    """

    print('handle points: ', handle_points)
    print('target points: ', target_points)

    assert len(handle_points) == len(target_points), \
        "number of handle point must equals target points"
    if text_embeddings is None:
        text_embeddings = model.get_text_embeddings(args.prompt)

    # the init output feature of unet
    with torch.no_grad():
        unet_output, F0 = model.forward_unet_features(
            init_code, 
            t,
            encoder_hidden_states=text_embeddings,
            layer_idx=args.unet_feature_idx,
            interp_res_h=args.sup_res_h,
            interp_res_w=args.sup_res_w,
        )
        # F0 is torch.Size([1, 640, 256, 256])
        x_prev_0, _ = model.step(unet_output, t, init_code)
        # init_code_orig = copy.deepcopy(init_code)

    # prepare optimizable init_code and optimizer
    init_code.requires_grad_(True)
    optimizer = torch.optim.Adam([init_code], lr=args.lr)
    opt_seq = [init_code.detach().clone()]

    # prepare for point tracking and background regularization
    handle_points_init = copy.deepcopy(handle_points)
    interp_mask = F.interpolate(mask, (init_code.shape[2],init_code.shape[3]), mode='nearest')
    using_mask = interp_mask.sum() != 0.0

    # prepare amp scaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()
    for step_idx in range(args.n_pix_step):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            unet_output, F1 = model.forward_unet_features(
                init_code, t,
                encoder_hidden_states=text_embeddings,
                layer_idx=args.unet_feature_idx,
                interp_res_h=args.sup_res_h,
                interp_res_w=args.sup_res_w,
            )
            # F1 is tensor ([1, 640, 256, 256])
            x_prev_updated, _ = model.step(unet_output, t, init_code)

            # do point tracking to update handle points before computing motion supervision loss
            if step_idx != 0:
                handle_points = point_tracking(F0, F1, handle_points, handle_points_init, args)
                print('new handle points', handle_points)

            # break if all handle points have reached the targets
            if check_handle_reach_target(handle_points, target_points):
                break

            loss = 0.0
            _, _, max_r, max_c = F0.shape
            patch_shapes = []
            for i in range(len(handle_points)):
                pi, ti = handle_points[i], target_points[i]
                # skip if the distance between target and source is less than 1
                if (ti - pi).norm() < 2.:
                    continue

                di = (ti - pi) / (ti - pi).norm()

                # motion supervision
                # with boundary protection
                # select row and column around p_i
                # r1, r2 = max(0, int(pi[0]) - args.r_m), min(max_r, int(pi[0]) + args.r_m+1)
                # c1, c2 = max(0, int(pi[1]) - args.r_m), min(max_c, int(pi[1]) + args.r_m+1)
                r1, r2 = max(0, int(pi[0]) - args.rects[i][0]), min(max_r, int(pi[0]) + args.rects[i][0] + 1)
                c1, c2 = max(0, int(pi[1]) - args.rects[i][1]), min(max_c, int(pi[1]) + args.rects[i][1] + 1)
                # slice F1 around handle point to get a term for eq (3)
                # sg(F_q(z^k_t))
                f0_patch = F1[:, :, r1:r2, c1:c2].detach()
                # q + d_i is not on the grid, interpolate patch of the same size
                f1_patch = interpolate_feature_patch(
                    feat=F1,
                    y1=r1+di[0],
                    y2=r2+di[0],
                    x1=c1+di[1],
                    x2=c2+di[1],
                )
                # f0 has shape [1, n_features, 3, 3], same as f0_patch

                # original code, without boundary protection
                # f0_patch = F1[:,:,int(pi[0])-args.r_m:int(pi[0])+args.r_m+1, int(pi[1])-args.r_m:int(pi[1])+args.r_m+1].detach()
                # f1_patch = interpolate_feature_patch(F1, pi[0] + di[0], pi[1] + di[1], args.r_m)
                loss += ((2*args.r_m+1)**2)*F.l1_loss(f0_patch, f1_patch)
                patch_shapes.append(f0_patch.shape)

            # masked region must stay unchanged
            if using_mask:
                loss += args.lam * ((x_prev_updated-x_prev_0)*(1.0-interp_mask)).abs().sum()
            # loss += args.lam * ((init_code_orig-init_code)*(1.0-interp_mask)).abs().sum()
            print(f'loss total={loss.item()} on patches {patch_shapes}')

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        opt_seq.append(init_code.detach().clone())

    return init_code, opt_seq


def run_drag(
        source_image,  # 512,512,3 numpy image [0,255]
        image_with_clicks,  # 512 512 3, used only to store results
        mask,  # 512, 512 numpy mask, 1 is inside
        prompt,
        points,  # list of pixel coordinates of [[handle], [target], [h2],[t2]] points
        inversion_strength=0.7,  # (0,1) - at which level optimization happens
        lam=0.1,  # opt parameter, weight at mask term
        latent_lr=0.01,
        n_pix_step=80,  # n optimization steps
        model_path="runwayml/stable-diffusion-v1-5",  # "runwayml/stable-diffusion-v1-5"
        vae_path="default",  # "default"
        lora_path="./lora_tmp",  # "./lora_tmp"
        start_step=0,  # used in MutualSelfAttentionControl only -> the step to start mutual self-attention control
        start_layer=10,  # used in MutualSelfAttentionControl only -> the layer to start mutual self-attention control
        save_dir="./results",
        save_seq=True,
        handle_whs: list = None,
        ):

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

    args.r_m = 1  # r1 in eq 3 (motion supervision)
    args.r_p = 3  # r2 in eq 5 (point tracking)
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
    # here, the point is in x,y pixel coordinate on the full resolution image
    # we transform them to resolution (sup_res_h, sup_res_w) for optimization
    # note that we also flip it x, y coordinates and store (y, x)
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

    handle_rect_whs = [[int(args.r_m), int(args.r_m)] for p in handle_points]
    if handle_whs is not None:
        for idx, wh in enumerate(handle_whs):
            handle_rect_whs[idx] = [
                int(wh[1] / full_h * args.sup_res_h),
                int(wh[0] / full_w * args.sup_res_w),
                ]
    print("handle_rect_whs: ", handle_rect_whs)
    args.rects = handle_rect_whs

    # set lora
    if lora_path == "":
        print("applying default parameters")
        # Disables custom attention processors and sets the default attention implementation.
        # see Unet2DConditionModel.set_default_attn_processor()
        model.unet.set_default_attn_processor()
        pass
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


configs = {
    "portrait42": {
        "image_path": "/home/ivan/projects/dreamslicer/data/portrait42/image.png",
        "json_path": "/home/ivan/projects/dreamslicer/data/portrait42/paper_theater_data.json",
        "default_inputs_path": 'inputs_portrait.pkl',
        "known_id": 9,
        "new_id": 7,
        "lora_path": "lora_portrait/",
    },
    "head": {
        "image_path": "dragbench_head.png",
        "json_path": "head.json",
        "default_inputs_path": 'inputs_head.pkl',
        "known_id": 0,
        "new_id": 1,
        "lora_path": "lora_head/",
    }
}


if __name__ == "__main__":

    name = "portrait42"
    # name = "head"
    conf = configs[name]

    with open(conf["default_inputs_path"], 'rb') as f:
        inputs = pickle.load(f)
    inputs['save_seq'] = False

    # # run default from pickle file
    # run_drag(
    #     **inputs,
    # )

    points_l_512, sizes_l_512 = load_points_from_theater_json(
        conf["json_path"],
        canvas_hw=(512, 512),
        known_scene_id=conf["known_id"],
        novel_scene_id=conf["new_id"],
    )

    print("sizes_l_512: ", sizes_l_512)

    inpimg, input_drawings = load_and_draw(
        conf["image_path"],
        points_list=points_l_512,
        sizes_list=sizes_l_512,
        return_type="np",
    )

    print("----------")

    print(inputs.keys())
    print(inputs["points"])
    print("-- > points from json")
    print(points_l_512)

    print("----------")

    out_img = run_drag(
        source_image=inpimg,
        image_with_clicks=input_drawings,
        mask=inputs['mask'],
        prompt="",
        points=points_l_512,
        save_seq=False,
        n_pix_step=80,
        lora_path=conf["lora_path"],
        # lora_path="",
        handle_whs=sizes_l_512,
    )
    # Convert the numpy array to a PIL Image
    pil_out_img = Image.fromarray(out_img)

    _, output_drawings = draw_on_image(pil_out_img, points_l_512, sizes_l_512, draw_rect_around="target")

    input_with_mask = overlap_mask(Image.fromarray(inpimg), mask=inputs['mask'])

    out_save = get_concat_h(
        get_concat_h(
            input_with_mask,
            Image.fromarray(input_drawings)
        ),
        Image.fromarray(output_drawings),
    )
    out_save.save(f"out_{name}.png")
    
    
    
