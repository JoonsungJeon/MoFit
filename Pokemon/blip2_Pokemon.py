import torch
# from pipeline_stable_diffusion import LocalStableDiffusionPipeline
# from diffusers import DDIMScheduler
import random
from utils import *
import argparse
import os
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torch.optim.adam import Adam
from torch.optim import SGD
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
import gc


from src.diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from src.diffusers import DDIMScheduler

from torch.optim import SGD
from torch.optim import RMSprop
from torch.optim import Adagrad
from lion_pytorch import Lion





def load_image(path, size=(512, 512)):
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    return np.array(img).astype(np.float32) / 255.0

def main_text_perimg(args):
    def load_pipeline(ckpt_path, device='cuda:0'):
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe
    
    
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "runwayml/stable-diffusion-v1-5"

    ##############################################################
    ckpt_path = "/mnt/nas5/joonsung/2025/ckpts/sd-pokemon-checkpoint/sd-pokemon-checkpoint"
    # ckpt_path = 'runwayml/stable-diffusion-v1-5'
    args.ckpt_path = ckpt_path

    
    
    pipe = load_pipeline(args.ckpt_path, device)
    ##############################################################
    set_random_seed(args.gen_seed)

    


    
    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    
    image = Image.open(args.img_path).convert("RGB")
    images = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 512, 512)
    
    prompt_id = args.img_path.split('/')[-1].split('.')[0]
    
    


        

    mse_loss = nn.MSELoss()
    
    garbage_prompt = ""
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[1].unsqueeze(0).detach()
    
    
    ################
    anchor = args.anchor
    until = args.until
    
    ## blip2
    if args.init == "ori":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/member_captions.jsonl"
        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/non_member_captions.jsonl"
        else:
             ValueError("args.mem was not satisfied.")
             
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["filename"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )
    
        init = args.init
        
    
    ## ori - mem
    elif args.init == "clip_interrogator":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_members_caption_output.jsonl"

        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_non_members_caption_output.jsonl"
            
            
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["prompt_id"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )

        init = args.init

        
        
    else:
        ValueError("args.init was not satisfied.")

    
    ################

    
    # noise_scale = 0.1
    iters = args.iters
    # eps = 0.5
    step_size = 0.25
    GUIDANCE_SCALE = 1.0

    ################
    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)

    timesteps = list(range(0, 200, 10))
        
    images = images.to(device)
    images = images*2. - 1.
    

    set_random_seed(args.gen_seed)
    
    rnd_text_emb = init_text_embeddings[1].unsqueeze(0).detach() # (1, 77, 768)
    
    with torch.no_grad():
        latent = pipe.vae.encode(images)
        latent = 0.18215 * latent.latent_dist.sample()

        # inverted_latents = invert(pipe, anchor, latent, garbage_prompt, device=device, guidance_scale=7.5, num_inference_steps=50)
        inverted_latents = pipe.inversion(prompt=None, latents=latent, text_embeddings=rnd_text_emb) ## conditional DDIM inversion

    latent_cur = inverted_latents[anchor+1]
    latent_prev = inverted_latents[anchor]

    timestep = timesteps[anchor+1]
    # timestep = torch.tensor([timestep], device=device)
    
    next_timestep = timesteps[anchor]
    # next_timestep = torch.tensor([next_timestep], device=device)
    
    # rnd_text_emb.requires_grad_(True)
    rnd_text_emb = rnd_text_emb.detach().clone().requires_grad_(True)
    ### From Null-text inversion ###
    # optimizer = Adam([rnd_text_emb], lr=1e-2 * (1. - t / 100.))
    optimizer = Adam([rnd_text_emb], lr=1e-2)


    def get_noise_pred_single(pipe, latents, t, context):
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=context).sample
        # noise_pred = pipe.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    with torch.no_grad():
        noise_pred_uncond = get_noise_pred_single(pipe, latent_cur, timestep, uncond_emb)
        
    
    
    pbar = tqdm(range(iters))
    for it in pbar:
        
        
        noise_pred_cond = get_noise_pred_single(pipe, latent_cur, timestep, rnd_text_emb)

        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)

        latents_prev_rec = pipe.scheduler.step(noise_pred, timestep, next_timestep, latent_cur, **extra_step_kwargs).prev_sample
        
    
        loss = (mse_loss(latents_prev_rec, latent_prev)).to(pipe.device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ## Early Stopping ##
        # loss_item = loss.item()
        # if loss_item < epsilon + t * 2e-5:
        #     break
        
        if pbar is not None:
            pbar.set_description(
                f"Image:{prompt_id} | Optimizing: t={timestep}->{next_timestep} | Iter {it} - Current loss: {loss.item():.8f}"
            )
        
    # cond_embeddings_list.append(rnd_text_emb[:1].detach())/
    
    
    rnd_text_emb = rnd_text_emb.detach().cpu()
    
    rnd_text_emb_npy = rnd_text_emb.numpy()    
    
     ## Save .npy
    if args.mem == "member":
        save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/SDv1_5_ver{args.ver}_anchor{anchor}_init_{init}_iters{iters}/members/perturb_emb"
    elif args.mem == "non_member":
        save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/SDv1_5_ver{args.ver}_anchor{anchor}_init_{init}_iters{iters}/non_members/perturb_emb"
    else:
        ValueError("args.mem was not satisfied.")
    
    
    # 
    
    os.makedirs(save_dir, exist_ok=True)
    np.save(save_dir+f"/{prompt_id}_anchor{anchor}_iter{iters}_{init}_Adam.npy", rnd_text_emb_npy)

    
        


        

        
def main_adv_text_per_img_1(args):
    
    
    
    
    def encode_prompt_(caption, tokenizer, text_encoder):
        captions = [caption]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(text_encoder.device)

        encoder_hidden_states = text_encoder(input_ids)[0]
        
        return encoder_hidden_states
    
    def load_pipeline(ckpt_path, device='cuda:0'):
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe
    
    
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "runwayml/stable-diffusion-v1-5"

    ##############################################################
    # ckpt_path = "/mnt/nas5/joonsung/2025/ckpts/sd-pokemon-checkpoint/sd-pokemon-checkpoint"
    
    # ckpt_path = 'runwayml/stable-diffusion-v1-5'
    

    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.ckpt_path, subfolder="tokenizer", revision=None
    # )
    # # tokenizer = tokenizer.to(device)
    # # tokenizer = tokenizer.cuda()

    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.ckpt_path, subfolder="text_encoder", revision=None
    # )
    # text_encoder = text_encoder.to(device)

    # vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    # vae = vae.to(device)

    # unet = UNet2DConditionModel.from_pretrained(
    #     args.ckpt_path, subfolder="unet", revision=None
    # )
    # unet = unet.to(device)
    
    # text_encoder.requires_grad_(False)
    
    # for p in text_encoder.parameters():
    #     p.requires_grad = False
    
    pipe = load_pipeline(args.ckpt_path, device)
    ##############################################################
    set_random_seed(args.gen_seed)


    

    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    
    image = Image.open(args.img_path).convert("RGB")
    images = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 512, 512)
    
    prompt_id = args.img_path.split('/')[-1].split('.')[0]
    
    


        

    mse_loss = nn.MSELoss()
    
    garbage_prompt = ""
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[1].unsqueeze(0).detach()
    
    
    
    ###################################
    
    
    anchor = args.anchor
    until = args.until
    


    ## blip2
    if args.init == "ori":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/member_captions.jsonl"
        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/non_member_captions.jsonl"
        else:
             ValueError("args.mem was not satisfied.")
             
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["filename"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )
    
        init = args.init
        
    
    ## ori - mem
    elif args.init == "clip_interrogator":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_members_caption_output.jsonl"

        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_non_members_caption_output.jsonl"
            
            
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["prompt_id"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )

        init = args.init

        
        
    else:
        ValueError("args.init was not satisfied.")

    
    # noise_scale = 0.1
    optim_iters = args.OptimIter
    iters = args.iters
    eps = args.eps
    step_size = eps/2.
    GUIDANCE_SCALE = 7.5
    # epsilon = 1e-5
    # step_size > 0.01: loss increase
    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
    
    # for p in pipe.text_encoder.parameters():
    #     p.requires_grad = False
    

    
    
    
    #### 1. Adv Example ####
    images = images*2. - 1.
    
    set_random_seed(args.gen_seed)
    adv_img = images.clone().detach() + (torch.rand(*images.shape)*2*eps-eps).to(device=device, dtype=torch.float32)

    
    
    
    
    ## 1. Uncond. DDIM inversion ##
    # set_random_seed(args.gen_seed)
    # with torch.no_grad():
    #     inverted_latents = invert(pipe, anchor, latent, garbage_prompt, device=device, guidance_scale=0, num_inference_steps=50)
        
    ## 2. add_noise ##
    # at the below FOR loop
    num_inference_steps = 100 ## SecMI setting

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    # print(pipe.scheduler.timesteps)
    
    timesteps = list(range(0, 200, 10))
    
    # trg_noise = torch.randn(inverted_latent.shape).to(device=device, dtype=torch.float32)
    

    
    # for p in pipe.unet.parameters():
    #     print(p.requires_grad) ## all True
        
    
    pbar_adv = tqdm(range(optim_iters))

    
    
    for t in range(anchor, until, -1):
        set_random_seed(args.gen_seed)
        # adv_img.requires_grad_(True)
        
        
        # optimizer = Adam([adv_img], lr=0.001) # 0.001
        # # optimizer = SGD([rnd_text_emb], lr=0.1) ## xxx 
        
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                             lr_lambda=lambda epoch: 0.95 ** epoch,
        #                             last_epoch=-1,
        #                             verbose=False)


        
        # timestep = pipe.scheduler.timesteps[50-t]
        
        timestep = timesteps[t]
        timestep = torch.tensor([timestep], device=device)
        
        
        for it in pbar_adv:
            set_random_seed(args.gen_seed)

            adv_img = adv_img.detach().clone().requires_grad_(True)
            
            
            ## 2. ##
            pipe.unet.zero_grad()
            pipe.vae.zero_grad()
            
            actual_step_size = step_size - (step_size - step_size / 100) / optim_iters * it
            # adv_latent_x0 = encode_image_grad(pipe, adv_img, dtype=torch.float32)
            
            adv_latent_x0 = pipe.vae.encode(adv_img.to(dtype=torch.float32))
            adv_latent_x0 = 0.18215 * adv_latent_x0.latent_dist.sample()

            rnd_noise = torch.randn(adv_latent_x0.shape).to(device=device, dtype=adv_latent_x0.dtype)
            
            adv_latent_xt = pipe.scheduler.add_noise(adv_latent_x0.to(device), rnd_noise.to(device), timestep)


            _, noise_pred = pipe.mtcnp_adv(perturb_embeds=None, perturb_latent=adv_latent_xt, prompt=pred_prompt, anchor=t, guidance_scale=7.5) ## default: 7.5


            

            ## 1. AdvPaint ##
            pipe.unet.zero_grad()
            cost = mse_loss(rnd_noise, noise_pred) / (anchor-until)
            grad, = torch.autograd.grad(cost, [adv_img])
            adv_img = adv_img - grad.sign() * actual_step_size
            adv_img = torch.minimum(torch.maximum(adv_img, adv_img - eps), adv_img + eps)
            adv_img.data = torch.clamp(adv_img, min=-1, max=1)
            adv_img.grad = None
            #### torch.cuda.empty_cache()

            ## 2. ## ==> ldm에서는 터짐
            # cost = (mse_loss(rnd_noise, noise_pred) / (anchor-until)).to(pipe.device)
            # cost.backward()
            # grad = adv_img.grad.detach().sign()
            # adv_img = adv_img - actual_step_size*grad
            # eta = torch.clamp(adv_img.data - images.data, min=-eps, max=eps)
            # adv_img = adv_img.detach()
            # adv_img = torch.clamp(adv_img + eta, min=-1, max=1)
            
            
            

            if pbar_adv is not None:
                pbar_adv.set_description(
                    f"Image: {prompt_id} | timestep {timestep.item()} | Iter {it} | eps {eps} --> Step size: {actual_step_size:.4f} / Current loss: {cost.item():.6f}"
                )


            ## 2. Adam
            # latent = encode_image(pipe, adv_img, dtype=torch.float32)
            # rnd_noise = torch.randn(latent.shape).to(device=device, dtype=latent.dtype)
            # latent_cur = pipe.scheduler.add_noise(latent.to(device), rnd_noise.to(device), timestep.to(device))

            # noise_pred_cond = get_noise_pred_single(pipe, latent_cur, timestep, rnd_text_emb)


        
            

        
    
    # torch.cuda.empty_cache()
    # del grad, adv_latent_x0, adv_latent_xt, noise_pred, cost 
    
    for var in [grad, adv_latent_x0, adv_latent_xt, noise_pred, cost]:
        if isinstance(var, torch.Tensor):
            var.detach_()
        del var

    torch.cuda.empty_cache()
    gc.collect()
    
    adv_img = adv_img.detach()
    

    adv_img_cpu = adv_img.cpu().squeeze(0)  # shape: (3, H, W)
    adv_img_cpu = (adv_img_cpu + 1) / 2  # Map to [0, 1]
    adv_img_pil = to_pil_image(adv_img_cpu)

    ## Save as PNG
    # if args.mem == "member":
    #     img_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/SDv1_5_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/adv_img"
    # elif args.mem == "non_member":
    #     img_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/SDv1_5_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/adv_img"
    # else:
    #     ValueError("args.mem was not satisfied.")

    # os.makedirs(img_dir, exist_ok=True)
    # save_path = os.path.join(img_dir, f"{prompt_id}_anchor{anchor}_OptimIter{optim_iters}_eps{eps}_step{step_size}.png")
    # adv_img_pil.save(save_path)
    
    
    
    
    #### 2. Text embedding ####
    ## New
    set_random_seed(args.gen_seed)
    
    rnd_text_emb = init_text_embeddings[1].unsqueeze(0).detach() # (1, 77, 768)
    
    with torch.no_grad():
        latent = pipe.vae.encode(adv_img)
        latent = 0.18215 * latent.latent_dist.sample()


        inverted_latents = pipe.inversion(prompt=None, latents=latent, text_embeddings=rnd_text_emb, guidance_scale=1.) ## default: g=1.0





    latent_cur = inverted_latents[anchor+1]
    latent_prev = inverted_latents[anchor]

    timestep = timesteps[anchor+1]
    timestep = torch.tensor([timestep], device=device)
    
    next_timestep = timesteps[anchor]
    next_timestep = torch.tensor([next_timestep], device=device)
    
    # rnd_text_emb.requires_grad_(True)
    rnd_text_emb = rnd_text_emb.detach().clone().requires_grad_(True)
    ### From Null-text inversion ###
    # optimizer = Adam([rnd_text_emb], lr=1e-2 * (1. - t / 100.))
    optimizer = Adam([rnd_text_emb], lr=1e-2)
    
    

    def get_noise_pred_single(pipe, latents, t, context):
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=context).sample
        # noise_pred = pipe.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    with torch.no_grad():
        noise_pred_uncond = get_noise_pred_single(pipe, latent_cur, timestep, uncond_emb)
        
    
    
    pbar = tqdm(range(iters))
    for it in pbar:
        
        
        noise_pred_cond = get_noise_pred_single(pipe, latent_cur, timestep, rnd_text_emb)

        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)

        latents_prev_rec = pipe.scheduler.step(noise_pred, timestep, next_timestep, latent_cur, **extra_step_kwargs).prev_sample
        
    
        loss = (mse_loss(latents_prev_rec, latent_prev)).to(pipe.device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ## Early Stopping ##
        # loss_item = loss.item()
        # if loss_item < epsilon + t * 2e-5:
        #     break
        
        if pbar is not None:
            pbar.set_description(
                f"Image:{prompt_id} | Optimizing: t={timestep.item()}->{next_timestep.item()} | Iter {it} - Current loss: {loss.item():.8f}"
            )
        
    # cond_embeddings_list.append(rnd_text_emb[:1].detach())/
    
    
    rnd_text_emb = rnd_text_emb.detach().cpu()
    
    rnd_text_emb_npy = rnd_text_emb.numpy()    


    ## Save .npy
    # if args.mem == "member":
    #     save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/SDv1_5_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/perturb_emb"
    # elif args.mem == "non_member":
    #     save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/SDv1_5_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/perturb_emb"
    # else:
    #     ValueError("args.mem was not satisfied.")

    # os.makedirs(save_dir, exist_ok=True)
    # np.save(save_dir+f"/{prompt_id}_anchor{anchor}_iter{iters}_{init}_Adam.npy", rnd_text_emb_npy)
    

    
    for var in [adv_img, rnd_text_emb, latent_cur, latent, latents_prev_rec, latent_prev, init_text_embeddings]:
        if isinstance(var, torch.Tensor):
            var.detach().clone()
        del var

    del optimizer 
    torch.cuda.empty_cache()
    gc.collect()
    

def main_adv_text_per_img_2(args):
    
    
    
    
    def encode_prompt_(caption, tokenizer, text_encoder):
        captions = [caption]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(text_encoder.device)

        encoder_hidden_states = text_encoder(input_ids)[0]
        
        return encoder_hidden_states
    
    def load_pipeline(ckpt_path, device='cuda:0'):
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe
    
    
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "runwayml/stable-diffusion-v1-5"

    ##############################################################
    ckpt_path = "/mnt/nas5/joonsung/2025/ckpts/sd-pokemon-checkpoint/sd-pokemon-checkpoint"
    # ckpt_path = 'runwayml/stable-diffusion-v1-5'
    args.ckpt_path = ckpt_path

    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.ckpt_path, subfolder="tokenizer", revision=None
    # )
    # # tokenizer = tokenizer.to(device)
    # # tokenizer = tokenizer.cuda()

    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.ckpt_path, subfolder="text_encoder", revision=None
    # )
    # text_encoder = text_encoder.to(device)

    # vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    # vae = vae.to(device)

    # unet = UNet2DConditionModel.from_pretrained(
    #     args.ckpt_path, subfolder="unet", revision=None
    # )
    # unet = unet.to(device)
    
    # text_encoder.requires_grad_(False)
    
    # for p in text_encoder.parameters():
    #     p.requires_grad = False
    
    pipe = load_pipeline(args.ckpt_path, device)
    ##############################################################
    set_random_seed(args.gen_seed)


    

    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    
    image = Image.open(args.img_path).convert("RGB")
    images = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 512, 512)
    
    prompt_id = args.img_path.split('/')[-1].split('.')[0]
    
    


        

    mse_loss = nn.MSELoss()
    
    garbage_prompt = ""
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[1].unsqueeze(0).detach()
    
    
    
    ###################################
    
    
    anchor = args.anchor
    until = args.until
    


    ## blip2
    if args.init == "ori":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/member_captions.jsonl"
        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/non_member_captions.jsonl"
        else:
             ValueError("args.mem was not satisfied.")
             
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["filename"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )
    
        init = args.init
        
    
    ## ori - mem
    elif args.init == "clip_interrogator":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_members_caption_output.jsonl"

        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_non_members_caption_output.jsonl"
            
            
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["prompt_id"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )

        init = args.init

        
        
    else:
        ValueError("args.init was not satisfied.")

    
    # noise_scale = 0.1
    optim_iters = args.OptimIter
    iters = args.iters
    eps = args.eps
    step_size = eps/2.
    GUIDANCE_SCALE = 1.0
    # epsilon = 1e-5
    # step_size > 0.01: loss increase
    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
    
    # for p in pipe.text_encoder.parameters():
    #     p.requires_grad = False
    

    
    
    
    #### 1. Adv Example ####
    images = images*2. - 1.
    
    set_random_seed(args.gen_seed)
    adv_img = images.clone().detach() + (torch.rand(*images.shape)*2*eps-eps).to(device=device, dtype=torch.float32)

    
    
    
    
    ## 1. Uncond. DDIM inversion ##
    # set_random_seed(args.gen_seed)
    # with torch.no_grad():
    #     inverted_latents = invert(pipe, anchor, latent, garbage_prompt, device=device, guidance_scale=0, num_inference_steps=50)
        
    ## 2. add_noise ##
    # at the below FOR loop
    num_inference_steps = 100 ## SecMI setting

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    # print(pipe.scheduler.timesteps)
    
    timesteps = list(range(0, 200, 10))
    
    # trg_noise = torch.randn(inverted_latent.shape).to(device=device, dtype=torch.float32)
    

    
    # for p in pipe.unet.parameters():
    #     print(p.requires_grad) ## all True
        
    
    pbar_adv = tqdm(range(optim_iters))

    
    
    for t in range(anchor, until, -1):
        set_random_seed(args.gen_seed)
        # adv_img.requires_grad_(True)
        
        
        # optimizer = Adam([adv_img], lr=0.001) # 0.001
        # # optimizer = SGD([rnd_text_emb], lr=0.1) ## xxx 
        
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                             lr_lambda=lambda epoch: 0.95 ** epoch,
        #                             last_epoch=-1,
        #                             verbose=False)


        
        # timestep = pipe.scheduler.timesteps[50-t]
        
        ## -------------------- old -------------------- ##
        timestep = timesteps[t]
        timestep = torch.tensor([timestep], device=device)
        ## -------------------- old -------------------- ##
        
        for it in pbar_adv:
            

            adv_img = adv_img.detach().clone().requires_grad_(True)

            # ## -------------------- NEW -------------------- ##
            # rand_t = torch.randint(0, t + 1, (1,)).item()  # 0 ~ t 사이의 랜덤 정수

            # # timesteps에서 해당 값 추출
            # timestep = timesteps[rand_t]
            # timestep = torch.tensor([timestep], device=device)

            # ## -------------------- NEW -------------------- ##
                        
            
            ## 2. ##
            pipe.unet.zero_grad()
            pipe.vae.zero_grad()
            
            actual_step_size = step_size - (step_size - step_size / 100) / optim_iters * it
            # adv_latent_x0 = encode_image_grad(pipe, adv_img, dtype=torch.float32)
            
            adv_latent_x0 = pipe.vae.encode(adv_img.to(dtype=torch.float32))
            adv_latent_x0 = 0.18215 * adv_latent_x0.latent_dist.sample()

            rnd_noise = torch.randn(adv_latent_x0.shape).to(device=device, dtype=adv_latent_x0.dtype)
            
            adv_latent_xt = pipe.scheduler.add_noise(adv_latent_x0.to(device), rnd_noise.to(device), timestep)


            _, noise_pred = pipe.mtcnp_adv(perturb_embeds=None, perturb_latent=adv_latent_xt, prompt=pred_prompt, anchor=t, guidance_scale=GUIDANCE_SCALE) ## default: 7.5


            

            ## 1. AdvPaint ##
            pipe.unet.zero_grad()
            cost = mse_loss(rnd_noise, noise_pred) / (anchor-until)
            grad, = torch.autograd.grad(cost, [adv_img])
            adv_img = adv_img - grad.sign() * actual_step_size
            adv_img = torch.minimum(torch.maximum(adv_img, adv_img - eps), adv_img + eps)
            adv_img.data = torch.clamp(adv_img, min=-1, max=1)
            adv_img.grad = None
            #### torch.cuda.empty_cache()

            ## 2. ## ==> ldm에서는 터짐
            # cost = (mse_loss(rnd_noise, noise_pred) / (anchor-until)).to(pipe.device)
            # cost.backward()
            # grad = adv_img.grad.detach().sign()
            # adv_img = adv_img - actual_step_size*grad
            # eta = torch.clamp(adv_img.data - images.data, min=-eps, max=eps)
            # adv_img = adv_img.detach()
            # adv_img = torch.clamp(adv_img + eta, min=-1, max=1)
            
            
            

            if pbar_adv is not None:
                pbar_adv.set_description(
                    f"Image: {prompt_id} | timestep {timestep.item()} | Iter {it} | eps {eps} --> Step size: {actual_step_size:.4f} / Current loss: {cost.item():.6f}"
                )


            ## 2. Adam
            # latent = encode_image(pipe, adv_img, dtype=torch.float32)
            # rnd_noise = torch.randn(latent.shape).to(device=device, dtype=latent.dtype)
            # latent_cur = pipe.scheduler.add_noise(latent.to(device), rnd_noise.to(device), timestep.to(device))

            # noise_pred_cond = get_noise_pred_single(pipe, latent_cur, timestep, rnd_text_emb)


        
            

        
    
    # torch.cuda.empty_cache()
    # del grad, adv_latent_x0, adv_latent_xt, noise_pred, cost 
    
    for var in [grad, adv_latent_x0, adv_latent_xt, noise_pred, cost]:
        if isinstance(var, torch.Tensor):
            var.detach_()
        del var

    torch.cuda.empty_cache()
    gc.collect()
    
    adv_img = adv_img.detach()
    

    adv_img_cpu = adv_img.cpu().squeeze(0)  # shape: (3, H, W)
    adv_img_cpu = (adv_img_cpu + 1) / 2  # Map to [0, 1]
    adv_img_pil = to_pil_image(adv_img_cpu)

    ## Save as PNG
    if args.mem == "member":
        img_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/SDv1_5_ver{args.ver}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/adv_img"
    elif args.mem == "non_member":
        img_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/SDv1_5_ver{args.ver}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/adv_img"
    else:
        ValueError("args.mem was not satisfied.")

    os.makedirs(img_dir, exist_ok=True)
    save_path = os.path.join(img_dir, f"{prompt_id}_anchor{anchor}_OptimIter{optim_iters}_eps{eps}_step{step_size}.png")
    adv_img_pil.save(save_path)
    
    
    
    
    #### 2. Text embedding ####
    ## New
    set_random_seed(args.gen_seed)
    
    rnd_text_emb = init_text_embeddings[1].unsqueeze(0).detach() # (1, 77, 768)
    
    with torch.no_grad():
        latent = pipe.vae.encode(adv_img)
        latent = 0.18215 * latent.latent_dist.sample()


        inverted_latents = pipe.inversion(prompt=None, latents=latent, text_embeddings=rnd_text_emb, guidance_scale=GUIDANCE_SCALE) ## default: g=1.0





    latent_cur = inverted_latents[anchor+1]
    latent_prev = inverted_latents[anchor]

    timestep = timesteps[anchor+1]
    timestep = torch.tensor([timestep], device=device)
    
    next_timestep = timesteps[anchor]
    next_timestep = torch.tensor([next_timestep], device=device)
    
    # rnd_text_emb.requires_grad_(True)
    rnd_text_emb = rnd_text_emb.detach().clone().requires_grad_(True)
    ### From Null-text inversion ###
    # optimizer = Adam([rnd_text_emb], lr=1e-2 * (1. - t / 100.))
    optimizer = Adam([rnd_text_emb], lr=1e-2)
    
    

    def get_noise_pred_single(pipe, latents, t, context):
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=context).sample
        # noise_pred = pipe.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    with torch.no_grad():
        noise_pred_uncond = get_noise_pred_single(pipe, latent_cur, timestep, uncond_emb)
        
    
    
    pbar = tqdm(range(iters))
    for it in pbar:
        
        
        noise_pred_cond = get_noise_pred_single(pipe, latent_cur, timestep, rnd_text_emb)

        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)

        latents_prev_rec = pipe.scheduler.step(noise_pred, timestep, next_timestep, latent_cur, **extra_step_kwargs).prev_sample
        
    
        loss = (mse_loss(latents_prev_rec, latent_prev)).to(pipe.device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ## Early Stopping ##
        # loss_item = loss.item()
        # if loss_item < epsilon + t * 2e-5:
        #     break
        
        if pbar is not None:
            pbar.set_description(
                f"Image:{prompt_id} | Optimizing: t={timestep.item()}->{next_timestep.item()} | Iter {it} - Current loss: {loss.item():.8f}"
            )
        
    # cond_embeddings_list.append(rnd_text_emb[:1].detach())/
    
    
    rnd_text_emb = rnd_text_emb.detach().cpu()
    
    rnd_text_emb_npy = rnd_text_emb.numpy()    


    ## Save .npy
    if args.mem == "member":
        save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/SDv1_5_ver{args.ver}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/perturb_emb"
    elif args.mem == "non_member":
        save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/SDv1_5_ver{args.ver}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/perturb_emb"
    else:
        ValueError("args.mem was not satisfied.")
    
    
    # 
    
    os.makedirs(save_dir, exist_ok=True)
    np.save(save_dir+f"/{prompt_id}_anchor{anchor}_iter{iters}_{init}_Adam.npy", rnd_text_emb_npy)
    

    
    for var in [adv_img, rnd_text_emb, latent_cur, latent, latents_prev_rec, latent_prev, init_text_embeddings]:
        if isinstance(var, torch.Tensor):
            var.detach().clone()
        del var

    del optimizer 
    torch.cuda.empty_cache()
    gc.collect()


def main_adv_text_per_img_3(args):
    
    
    
    def load_pipeline(ckpt_path, device='cuda:0'):
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe
    
    
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "runwayml/stable-diffusion-v1-5"

    ##############################################################
    ckpt_path = "/mnt/nas5/joonsung/2025/ckpts/sd-pokemon-checkpoint/sd-pokemon-checkpoint"
    # ckpt_path = 'runwayml/stable-diffusion-v1-5'
    args.ckpt_path = ckpt_path

    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.ckpt_path, subfolder="tokenizer", revision=None
    # )
    # # tokenizer = tokenizer.to(device)
    # # tokenizer = tokenizer.cuda()

    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.ckpt_path, subfolder="text_encoder", revision=None
    # )
    # text_encoder = text_encoder.to(device)

    # vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    # vae = vae.to(device)

    # unet = UNet2DConditionModel.from_pretrained(
    #     args.ckpt_path, subfolder="unet", revision=None
    # )
    # unet = unet.to(device)
    
    # text_encoder.requires_grad_(False)
    
    # for p in text_encoder.parameters():
    #     p.requires_grad = False
    
    pipe = load_pipeline(args.ckpt_path, device)
    ##############################################################
    set_random_seed(args.gen_seed)


    

    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    
    image = Image.open(args.img_path).convert("RGB")
    images = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 512, 512)
    
    prompt_id = args.img_path.split('/')[-1].split('.')[0]
    
    


        

    mse_loss = nn.MSELoss()
    
    garbage_prompt = ""
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[1].unsqueeze(0).detach()
    
    
    
    ###################################
    
    
    anchor = args.anchor
    until = args.until
    


    ## blip2
    if args.init == "ori":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/member_captions.jsonl"
        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/non_member_captions.jsonl"
        else:
             ValueError("args.mem was not satisfied.")
             
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["filename"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )
    
        init = args.init
        
    
    ## ori - mem
    elif args.init == "clip_interrogator":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_members_caption_output.jsonl"

        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_non_members_caption_output.jsonl"
            
            
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["prompt_id"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )

        init = args.init

        
        
    else:
        ValueError("args.init was not satisfied.")

    
    # noise_scale = 0.1
    optim_iters = args.OptimIter
    iters = args.iters
    eps = args.eps
    step_size = eps/2.
    GUIDANCE_SCALE = 1.0
    # epsilon = 1e-5
    # step_size > 0.01: loss increase
    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
    
    # for p in pipe.text_encoder.parameters():
    #     p.requires_grad = False
    

    
    
    
    #### 1. Adv Example ####
    images = images*2. - 1.
    
    set_random_seed(args.gen_seed)
    adv_img = images.clone().detach() + (torch.rand(*images.shape)*2*eps-eps).to(device=device, dtype=torch.float32)

    
    
    
    
    ## 1. Uncond. DDIM inversion ##
    # set_random_seed(args.gen_seed)
    # with torch.no_grad():
    #     inverted_latents = invert(pipe, anchor, latent, garbage_prompt, device=device, guidance_scale=0, num_inference_steps=50)
        
    ## 2. add_noise ##
    # at the below FOR loop
    num_inference_steps = 100 ## SecMI setting

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    # print(pipe.scheduler.timesteps)
    
    timesteps = list(range(0, 200, 10))
    
    # trg_noise = torch.randn(inverted_latent.shape).to(device=device, dtype=torch.float32)
    

    
    # for p in pipe.unet.parameters():
    #     print(p.requires_grad) ## all True
        
    
    pbar_adv = tqdm(range(optim_iters))

    
    
    for t in range(anchor, until, -1):
        set_random_seed(args.gen_seed)
        # adv_img.requires_grad_(True)
        
        
        # optimizer = Adam([adv_img], lr=0.001) # 0.001
        # # optimizer = SGD([rnd_text_emb], lr=0.1) ## xxx 
        
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                             lr_lambda=lambda epoch: 0.95 ** epoch,
        #                             last_epoch=-1,
        #                             verbose=False)


        
        # timestep = pipe.scheduler.timesteps[50-t]
        
        ## -------------------- old -------------------- ##
        # timestep = timesteps[t]
        # timestep = torch.tensor([timestep], device=device)
        ## -------------------- old -------------------- ##
        
        for it in pbar_adv:
            

            adv_img = adv_img.detach().clone().requires_grad_(True)

            ## -------------------- NEW -------------------- ##
            rand_t = torch.randint(0, t + 1, (1,)).item()  # 0 ~ t 사이의 랜덤 정수

            # timesteps에서 해당 값 추출
            timestep = timesteps[rand_t]
            timestep = torch.tensor([timestep], device=device)

            ## -------------------- NEW -------------------- ##
                        
            
            ## 2. ##
            pipe.unet.zero_grad()
            pipe.vae.zero_grad()
            
            actual_step_size = step_size - (step_size - step_size / 100) / optim_iters * it
            # adv_latent_x0 = encode_image_grad(pipe, adv_img, dtype=torch.float32)
            
            adv_latent_x0 = pipe.vae.encode(adv_img.to(dtype=torch.float32))
            adv_latent_x0 = 0.18215 * adv_latent_x0.latent_dist.sample()

            rnd_noise = torch.randn(adv_latent_x0.shape).to(device=device, dtype=adv_latent_x0.dtype)
            
            adv_latent_xt = pipe.scheduler.add_noise(adv_latent_x0.to(device), rnd_noise.to(device), timestep)


            _, noise_pred = pipe.mtcnp_adv(perturb_embeds=None, perturb_latent=adv_latent_xt, prompt=pred_prompt, anchor=rand_t, guidance_scale=GUIDANCE_SCALE) ## default: 7.5


            

            ## 1. AdvPaint ##
            pipe.unet.zero_grad()
            cost = mse_loss(rnd_noise, noise_pred) / (anchor-until)
            grad, = torch.autograd.grad(cost, [adv_img])
            adv_img = adv_img - grad.sign() * actual_step_size
            adv_img = torch.minimum(torch.maximum(adv_img, adv_img - eps), adv_img + eps)
            adv_img.data = torch.clamp(adv_img, min=-1, max=1)
            adv_img.grad = None
            #### torch.cuda.empty_cache()

            ## 2. ## ==> ldm에서는 터짐
            # cost = (mse_loss(rnd_noise, noise_pred) / (anchor-until)).to(pipe.device)
            # cost.backward()
            # grad = adv_img.grad.detach().sign()
            # adv_img = adv_img - actual_step_size*grad
            # eta = torch.clamp(adv_img.data - images.data, min=-eps, max=eps)
            # adv_img = adv_img.detach()
            # adv_img = torch.clamp(adv_img + eta, min=-1, max=1)
            
            
            

            if pbar_adv is not None:
                pbar_adv.set_description(
                    f"Image: {prompt_id} | timestep {timestep.item()} | Iter {it} | eps {eps} --> Step size: {actual_step_size:.4f} / Current loss: {cost.item():.6f}"
                )


            ## 2. Adam
            # latent = encode_image(pipe, adv_img, dtype=torch.float32)
            # rnd_noise = torch.randn(latent.shape).to(device=device, dtype=latent.dtype)
            # latent_cur = pipe.scheduler.add_noise(latent.to(device), rnd_noise.to(device), timestep.to(device))

            # noise_pred_cond = get_noise_pred_single(pipe, latent_cur, timestep, rnd_text_emb)


        
            

        
    
    # torch.cuda.empty_cache()
    # del grad, adv_latent_x0, adv_latent_xt, noise_pred, cost 
    
    for var in [grad, adv_latent_x0, adv_latent_xt, noise_pred, cost]:
        if isinstance(var, torch.Tensor):
            var.detach_()
        del var

    torch.cuda.empty_cache()
    gc.collect()
    
    adv_img = adv_img.detach()
    

    adv_img_cpu = adv_img.cpu().squeeze(0)  # shape: (3, H, W)
    adv_img_cpu = (adv_img_cpu + 1) / 2  # Map to [0, 1]
    adv_img_pil = to_pil_image(adv_img_cpu)

    ## Save as PNG
    if args.mem == "member":
        img_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/SDv1_5_ver{args.ver}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/adv_img"
    elif args.mem == "non_member":
        img_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/SDv1_5_ver{args.ver}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/adv_img"
    else:
        ValueError("args.mem was not satisfied.")

    os.makedirs(img_dir, exist_ok=True)
    save_path = os.path.join(img_dir, f"{prompt_id}_anchor{anchor}_OptimIter{optim_iters}_eps{eps}_step{step_size}.png")
    adv_img_pil.save(save_path)
    
    
    
    
    #### 2. Text embedding ####
    ## New
    set_random_seed(args.gen_seed)
    
    rnd_text_emb = init_text_embeddings[1].unsqueeze(0).detach() # (1, 77, 768)
    
    with torch.no_grad():
        latent = pipe.vae.encode(adv_img)
        latent = 0.18215 * latent.latent_dist.sample()


        inverted_latents = pipe.inversion(prompt=None, latents=latent, text_embeddings=rnd_text_emb, guidance_scale=GUIDANCE_SCALE) ## default: g=1.0





    latent_cur = inverted_latents[anchor+1]
    latent_prev = inverted_latents[anchor]

    timestep = timesteps[anchor+1]
    timestep = torch.tensor([timestep], device=device)
    
    next_timestep = timesteps[anchor]
    next_timestep = torch.tensor([next_timestep], device=device)
    
    # rnd_text_emb.requires_grad_(True)
    rnd_text_emb = rnd_text_emb.detach().clone().requires_grad_(True)
    ### From Null-text inversion ###
    # optimizer = Adam([rnd_text_emb], lr=1e-2 * (1. - t / 100.))
    optimizer = Adam([rnd_text_emb], lr=1e-2)
    
    

    def get_noise_pred_single(pipe, latents, t, context):
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=context).sample
        # noise_pred = pipe.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    with torch.no_grad():
        noise_pred_uncond = get_noise_pred_single(pipe, latent_cur, timestep, uncond_emb)
        
    
    
    pbar = tqdm(range(iters))
    for it in pbar:
        
        
        noise_pred_cond = get_noise_pred_single(pipe, latent_cur, timestep, rnd_text_emb)

        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)

        latents_prev_rec = pipe.scheduler.step(noise_pred, timestep, next_timestep, latent_cur, **extra_step_kwargs).prev_sample
        
    
        loss = (mse_loss(latents_prev_rec, latent_prev)).to(pipe.device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ## Early Stopping ##
        # loss_item = loss.item()
        # if loss_item < epsilon + t * 2e-5:
        #     break
        
        if pbar is not None:
            pbar.set_description(
                f"Image:{prompt_id} | Optimizing: t={timestep.item()}->{next_timestep.item()} | Iter {it} - Current loss: {loss.item():.8f}"
            )
        
    # cond_embeddings_list.append(rnd_text_emb[:1].detach())/
    
    
    rnd_text_emb = rnd_text_emb.detach().cpu()
    
    rnd_text_emb_npy = rnd_text_emb.numpy()    


    ## Save .npy
    if args.mem == "member":
        save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/SDv1_5_ver{args.ver}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/perturb_emb"
    elif args.mem == "non_member":
        save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/SDv1_5_ver{args.ver}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/perturb_emb"
    else:
        ValueError("args.mem was not satisfied.")
    
    
    # 
    
    os.makedirs(save_dir, exist_ok=True)
    np.save(save_dir+f"/{prompt_id}_anchor{anchor}_iter{iters}_{init}_Adam.npy", rnd_text_emb_npy)
    
    
    
    
def main_adv_text_per_img_5(args):
    
    
    
    def load_pipeline(ckpt_path, device='cuda:0'):
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe
    
    
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "runwayml/stable-diffusion-v1-5"

    ##############################################################
    ckpt_path = "/mnt/nas5/joonsung/2025/ckpts/sd-pokemon-checkpoint/sd-pokemon-checkpoint"
    # ckpt_path = 'runwayml/stable-diffusion-v1-5'
    args.ckpt_path = ckpt_path

    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.ckpt_path, subfolder="tokenizer", revision=None
    # )
    # # tokenizer = tokenizer.to(device)
    # # tokenizer = tokenizer.cuda()

    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.ckpt_path, subfolder="text_encoder", revision=None
    # )
    # text_encoder = text_encoder.to(device)

    # vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    # vae = vae.to(device)

    # unet = UNet2DConditionModel.from_pretrained(
    #     args.ckpt_path, subfolder="unet", revision=None
    # )
    # unet = unet.to(device)
    
    # text_encoder.requires_grad_(False)
    
    # for p in text_encoder.parameters():
    #     p.requires_grad = False
    
    pipe = load_pipeline(args.ckpt_path, device)
    ##############################################################
    set_random_seed(args.gen_seed)


    

    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    
    image = Image.open(args.img_path).convert("RGB")
    images = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 512, 512)
    
    prompt_id = args.img_path.split('/')[-1].split('.')[0]
    
    


        

    mse_loss = nn.MSELoss()
    
    garbage_prompt = ""
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[1].unsqueeze(0).detach()
    
    
    
    ###################################
    
    
    anchor = args.anchor
    until = args.until
    


    ## blip2
    if args.init == "ori":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/member_captions.jsonl"
        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/non_member_captions.jsonl"
        else:
             ValueError("args.mem was not satisfied.")
             
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["filename"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )
    
        init = args.init
        
    
    ## ori - mem
    elif args.init == "clip_interrogator":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_members_caption_output.jsonl"

        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_non_members_caption_output.jsonl"
            
            
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["prompt_id"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )

        init = args.init

        
        
    else:
        ValueError("args.init was not satisfied.")

    
    # noise_scale = 0.1
    optim_iters = args.OptimIter
    iters = args.iters
    eps = args.eps
    step_size = eps/2.
    GUIDANCE_SCALE = 1.0
    # epsilon = 1e-5
    # step_size > 0.01: loss increase
    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
    
    # for p in pipe.text_encoder.parameters():
    #     p.requires_grad = False
    

    
    
    
    #### 1. Adv Example ####
    images = images*2. - 1.
    
    set_random_seed(args.gen_seed)
    adv_img = images.clone().detach() + (torch.rand(*images.shape)*2*eps-eps).to(device=device, dtype=torch.float32)

    
    
    
    
    ## 1. Uncond. DDIM inversion ##
    # set_random_seed(args.gen_seed)
    # with torch.no_grad():
    #     inverted_latents = invert(pipe, anchor, latent, garbage_prompt, device=device, guidance_scale=0, num_inference_steps=50)
        
    ## 2. add_noise ##
    # at the below FOR loop
    num_inference_steps = 100 ## SecMI setting

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    # print(pipe.scheduler.timesteps)
    
    timesteps = list(range(0, 200, 10))
    
    # trg_noise = torch.randn(inverted_latent.shape).to(device=device, dtype=torch.float32)
    

    
    # for p in pipe.unet.parameters():
    #     print(p.requires_grad) ## all True
        
    
    pbar_adv = tqdm(range(optim_iters))

    
    
    for t in range(anchor, until, -1):
        set_random_seed(args.gen_seed)
        # adv_img.requires_grad_(True)
        
        
        # optimizer = Adam([adv_img], lr=0.001) # 0.001
        # # optimizer = SGD([rnd_text_emb], lr=0.1) ## xxx 
        
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                             lr_lambda=lambda epoch: 0.95 ** epoch,
        #                             last_epoch=-1,
        #                             verbose=False)


        
        # timestep = pipe.scheduler.timesteps[50-t]
        
        ## -------------------- old -------------------- ##
        # timestep = timesteps[t]
        # timestep = torch.tensor([timestep], device=device)
        ## -------------------- old -------------------- ##
        
        for it in pbar_adv:
            

            adv_img = adv_img.detach().clone().requires_grad_(True)

            ## -------------------- NEW -------------------- ##
            rand_t = torch.randint(0, t + 1, (1,)).item()  # 0 ~ t 사이의 랜덤 정수

            # timesteps에서 해당 값 추출
            timestep = timesteps[rand_t]
            timestep = torch.tensor([timestep], device=device)

            ## -------------------- NEW -------------------- ##
                        
            
            ## 2. ##
            pipe.unet.zero_grad()
            pipe.vae.zero_grad()
            
            actual_step_size = step_size - (step_size - step_size / 100) / optim_iters * it
            # adv_latent_x0 = encode_image_grad(pipe, adv_img, dtype=torch.float32)
            
            adv_latent_x0 = pipe.vae.encode(adv_img.to(dtype=torch.float32))
            adv_latent_x0 = 0.18215 * adv_latent_x0.latent_dist.sample()

            rnd_noise = torch.randn(adv_latent_x0.shape).to(device=device, dtype=adv_latent_x0.dtype)
            
            adv_latent_xt = pipe.scheduler.add_noise(adv_latent_x0.to(device), rnd_noise.to(device), timestep)


            _, noise_pred, noise_pred_uncond, noise_pred_text = pipe.mtcnp_adv(perturb_embeds=None, perturb_latent=adv_latent_xt, prompt=pred_prompt, anchor=rand_t, guidance_scale=7.5) ## default: 7.5


            

            ## 1. AdvPaint ##
            pipe.unet.zero_grad()
            ## ------------ NEW ------------ ##
            cost = mse_loss(rnd_noise, noise_pred_uncond) / (anchor-until)
            ## ------------ NEW ------------ ##
            grad, = torch.autograd.grad(cost, [adv_img])
            adv_img = adv_img - grad.sign() * actual_step_size
            adv_img = torch.minimum(torch.maximum(adv_img, adv_img - eps), adv_img + eps)
            adv_img.data = torch.clamp(adv_img, min=-1, max=1)
            adv_img.grad = None
            #### torch.cuda.empty_cache()

            ## 2. ## ==> ldm에서는 터짐
            # cost = (mse_loss(rnd_noise, noise_pred) / (anchor-until)).to(pipe.device)
            # cost.backward()
            # grad = adv_img.grad.detach().sign()
            # adv_img = adv_img - actual_step_size*grad
            # eta = torch.clamp(adv_img.data - images.data, min=-eps, max=eps)
            # adv_img = adv_img.detach()
            # adv_img = torch.clamp(adv_img + eta, min=-1, max=1)
            
            
            

            if pbar_adv is not None:
                pbar_adv.set_description(
                    f"Image: {prompt_id} | timestep {timestep.item()} | Iter {it} | eps {eps} --> Step size: {actual_step_size:.4f} / Current loss: {cost.item():.6f}"
                )


            ## 2. Adam
            # latent = encode_image(pipe, adv_img, dtype=torch.float32)
            # rnd_noise = torch.randn(latent.shape).to(device=device, dtype=latent.dtype)
            # latent_cur = pipe.scheduler.add_noise(latent.to(device), rnd_noise.to(device), timestep.to(device))

            # noise_pred_cond = get_noise_pred_single(pipe, latent_cur, timestep, rnd_text_emb)


        
            

        
    
    # torch.cuda.empty_cache()
    # del grad, adv_latent_x0, adv_latent_xt, noise_pred, cost 
    
    for var in [grad, adv_latent_x0, adv_latent_xt, noise_pred, cost]:
        if isinstance(var, torch.Tensor):
            var.detach_()
        del var

    torch.cuda.empty_cache()
    gc.collect()
    
    adv_img = adv_img.detach()
    

    adv_img_cpu = adv_img.cpu().squeeze(0)  # shape: (3, H, W)
    adv_img_cpu = (adv_img_cpu + 1) / 2  # Map to [0, 1]
    adv_img_pil = to_pil_image(adv_img_cpu)

    ## Save as PNG
    # if args.mem == "member":
    #     img_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/SDv1_5_ver{args.ver}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/adv_img"
    # elif args.mem == "non_member":
    #     img_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/SDv1_5_ver{args.ver}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/adv_img"
    # else:
    #     ValueError("args.mem was not satisfied.")

    # os.makedirs(img_dir, exist_ok=True)
    # save_path = os.path.join(img_dir, f"{prompt_id}_anchor{anchor}_OptimIter{optim_iters}_eps{eps}_step{step_size}.png")
    # adv_img_pil.save(save_path)
    
    
    
    
    
    
    #### 2. Text embedding ####
    ## New
    set_random_seed(args.gen_seed)
    
    rnd_text_emb = init_text_embeddings[1].unsqueeze(0).detach() # (1, 77, 768)
    
    
    # rnd_text_emb.requires_grad_(True)
    rnd_text_emb = rnd_text_emb.detach().clone().requires_grad_(True)
    ### From Null-text inversion ###
    # optimizer = Adam([rnd_text_emb], lr=1e-2 * (1. - t / 100.))
    optimizer = Adam([rnd_text_emb], lr=1e-2)
    
    

    def get_noise_pred_single(pipe, latents, t, context):
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=context).sample
        # noise_pred = pipe.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    
    pbar = tqdm(range(iters))
    
    
    timestep = timesteps[anchor]
    timestep = torch.tensor([timestep], device=device)
    for it in pbar:
        
        ## 1. ##
        # rand_t = torch.randint(0, t + 1, (1,)).item()  # 0 ~ t 사이의 랜덤 정수

        # # timesteps에서 해당 값 추출
        # timestep = timesteps[rand_t]
        # timestep = torch.tensor([timestep], device=device)

        ## ---------------------------------------- ##
                    
        
        ## 2. ##
        pipe.unet.zero_grad()
        pipe.vae.zero_grad()
        
        
        with torch.no_grad():
            adv_latent_x0 = pipe.vae.encode(adv_img.to(dtype=torch.float32))
            adv_latent_x0 = 0.18215 * adv_latent_x0.latent_dist.sample()

            rnd_noise = torch.randn(adv_latent_x0.shape).to(device=device, dtype=adv_latent_x0.dtype)
            
            adv_latent_xt = pipe.scheduler.add_noise(adv_latent_x0.to(device), rnd_noise.to(device), timestep)


        noise_pred_cond = get_noise_pred_single(pipe, adv_latent_xt, timestep, rnd_text_emb)
    
        loss = (mse_loss(rnd_noise, noise_pred_cond)).to(pipe.device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ## Early Stopping ##
        # loss_item = loss.item()
        # if loss_item < epsilon + t * 2e-5:
        #     break
        
        if pbar is not None:
            pbar.set_description(
                f"[Embedding] image: {prompt_id} | timestep: {timestep.item()} | Iter: {it} -- Current loss: {loss.item():.6f}"
            )
    
    # cond_embeddings_list.append(rnd_text_emb[:1].detach())/
    
    
    rnd_text_emb = rnd_text_emb.detach().cpu()
    
    rnd_text_emb_npy = rnd_text_emb.numpy()    


    # ## Save .npy
    # if args.mem == "member":
    #     save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/SDv1_5_ver{args.ver}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/perturb_emb"
    # elif args.mem == "non_member":
    #     save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/SDv1_5_ver{args.ver}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/perturb_emb"
    # else:
    #     ValueError("args.mem was not satisfied.")
    
    
    # # 
    
    # os.makedirs(save_dir, exist_ok=True)
    # np.save(save_dir+f"/{prompt_id}_anchor{anchor}_iter{iters}_{init}_Adam.npy", rnd_text_emb_npy)
    

    

 
def main_adv_text_per_img_7(args):
    
    
    
    
    def encode_prompt_(caption, tokenizer, text_encoder):
        captions = [caption]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(text_encoder.device)

        encoder_hidden_states = text_encoder(input_ids)[0]
        
        return encoder_hidden_states
    
    def load_pipeline(ckpt_path, device='cuda:0'):
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe
    
    
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "runwayml/stable-diffusion-v1-5"

    ##############################################################
    # ckpt_path = "/mnt/nas5/joonsung/2025/ckpts/sd-pokemon-checkpoint/sd-pokemon-checkpoint"
    
    # ckpt_path = 'runwayml/stable-diffusion-v1-5'
    

    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.ckpt_path, subfolder="tokenizer", revision=None
    # )
    # # tokenizer = tokenizer.to(device)
    # # tokenizer = tokenizer.cuda()

    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.ckpt_path, subfolder="text_encoder", revision=None
    # )
    # text_encoder = text_encoder.to(device)

    # vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    # vae = vae.to(device)

    # unet = UNet2DConditionModel.from_pretrained(
    #     args.ckpt_path, subfolder="unet", revision=None
    # )
    # unet = unet.to(device)
    
    # text_encoder.requires_grad_(False)
    
    # for p in text_encoder.parameters():
    #     p.requires_grad = False
    
    pipe = load_pipeline(args.ckpt_path, device)
    ##############################################################
    set_random_seed(args.gen_seed)


    

    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    
    image = Image.open(args.img_path).convert("RGB")
    images = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 512, 512)
    
    prompt_id = args.img_path.split('/')[-1].split('.')[0]
    
    


        

    mse_loss = nn.MSELoss()
    
    garbage_prompt = ""
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[1].unsqueeze(0).detach()
    
    
    
    ###################################
    
    
    anchor = args.anchor
    until = args.until
    


    ## blip2
    if args.init == "ori":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/member_captions.jsonl"
        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/non_member_captions.jsonl"
        else:
             ValueError("args.mem was not satisfied.")
             
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["filename"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )
    
        init = args.init
        
    
    ## ori - mem
    elif args.init == "clip_interrogator":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_members_caption_output.jsonl"

        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_non_members_caption_output.jsonl"
            
            
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["prompt_id"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )

        init = args.init

        
        
    else:
        ValueError("args.init was not satisfied.")

    
    # noise_scale = 0.1
    optim_iters = args.OptimIter
    iters = args.iters
    eps = args.eps
    step_size = eps/2.
    GUIDANCE_SCALE = 7.5
    # epsilon = 1e-5
    # step_size > 0.01: loss increase
    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
    
    # for p in pipe.text_encoder.parameters():
    #     p.requires_grad = False
    

    
    
    
    #### 1. Adv Example ####
    images = images*2. - 1.
    
    set_random_seed(args.gen_seed)
    adv_img = images.clone().detach() + (torch.rand(*images.shape)*2*eps-eps).to(device=device, dtype=torch.float32)

    
    
    
    
    ## 1. Uncond. DDIM inversion ##
    # set_random_seed(args.gen_seed)
    # with torch.no_grad():
    #     inverted_latents = invert(pipe, anchor, latent, garbage_prompt, device=device, guidance_scale=0, num_inference_steps=50)
        
    ## 2. add_noise ##
    # at the below FOR loop
    num_inference_steps = 50 ## SecMI setting

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    # print(pipe.scheduler.timesteps)
    
    timesteps = list(range(0, 1000, 20))
    
    # trg_noise = torch.randn(inverted_latent.shape).to(device=device, dtype=torch.float32)
    

    
    # for p in pipe.unet.parameters():
    #     print(p.requires_grad) ## all True
        
    
    pbar_adv = tqdm(range(optim_iters))

    
    
    for j in range(anchor, until, -1):
        set_random_seed(args.gen_seed)
        # adv_img.requires_grad_(True)
        
        
        # optimizer = Adam([adv_img], lr=0.001) # 0.001
        # # optimizer = SGD([rnd_text_emb], lr=0.1) ## xxx 
        
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                             lr_lambda=lambda epoch: 0.95 ** epoch,
        #                             last_epoch=-1,
        #                             verbose=False)


        
        ## -------------------- old -------------------- ##
        # t=j
        # timestep = timesteps[t]
        # timestep = torch.tensor([timestep], device=device)
        
        ## -------------------- old -------------------- ##
        
        ## anchor == 20
        
        for it in pbar_adv:
            set_random_seed(args.gen_seed)

            adv_img = adv_img.detach().clone().requires_grad_(True)
            
            
            ## -------------------- NEW -------------------- ##
            
            t = torch.randint(j-2, j + 3, (1,)).item()  # 0 ~ t 사이의 랜덤 정수
            
            ## anchor = 130 -> {110, 120, 130, 140, 150}

            # timesteps에서 해당 값 추출
            timestep = timesteps[t]
            timestep = torch.tensor([timestep], device=device)

            ## -------------------- NEW -------------------- ##  
            
            ## 2. ##
            pipe.unet.zero_grad()
            pipe.vae.zero_grad()
            
            actual_step_size = step_size - (step_size - step_size / 100) / optim_iters * it
            # adv_latent_x0 = encode_image_grad(pipe, adv_img, dtype=torch.float32)
            
            adv_latent_x0 = pipe.vae.encode(adv_img.to(dtype=torch.float32))
            adv_latent_x0 = 0.18215 * adv_latent_x0.latent_dist.sample()

            rnd_noise = torch.randn(adv_latent_x0.shape).to(device=device, dtype=adv_latent_x0.dtype)
            
            adv_latent_xt = pipe.scheduler.add_noise(adv_latent_x0.to(device), rnd_noise.to(device), timestep)


            _, _, noise_pred_uncond, noise_pred_text = pipe.mtcnp_adv(perturb_embeds=None, perturb_latent=adv_latent_xt, prompt=pred_prompt, anchor=t, guidance_scale=7.5) ## default: 7.5


            

            ## 1. AdvPaint ##
            pipe.unet.zero_grad()
            cost = mse_loss(noise_pred_uncond, noise_pred_text)
            grad, = torch.autograd.grad(cost, [adv_img])
            adv_img = adv_img + grad.sign() * actual_step_size
            adv_img = torch.minimum(torch.maximum(adv_img, adv_img - eps), adv_img + eps)
            adv_img.data = torch.clamp(adv_img, min=-1, max=1)
            adv_img.grad = None
            #### torch.cuda.empty_cache()

            ## 2. ## ==> ldm에서는 터짐
            # cost = (mse_loss(rnd_noise, noise_pred) / (anchor-until)).to(pipe.device)
            # cost.backward()
            # grad = adv_img.grad.detach().sign()
            # adv_img = adv_img - actual_step_size*grad
            # eta = torch.clamp(adv_img.data - images.data, min=-eps, max=eps)
            # adv_img = adv_img.detach()
            # adv_img = torch.clamp(adv_img + eta, min=-1, max=1)
            
            
            

            if pbar_adv is not None:
                pbar_adv.set_description(
                    f"Image: {prompt_id} | timestep {timestep.item()} | Iter {it} | eps {eps} --> Step size: {actual_step_size:.4f} / Current loss: {cost.item():.6f}"
                )


            ## 2. Adam
            # latent = encode_image(pipe, adv_img, dtype=torch.float32)
            # rnd_noise = torch.randn(latent.shape).to(device=device, dtype=latent.dtype)
            # latent_cur = pipe.scheduler.add_noise(latent.to(device), rnd_noise.to(device), timestep.to(device))

            # noise_pred_cond = get_noise_pred_single(pipe, latent_cur, timestep, rnd_text_emb)


        
            

        
    
    # torch.cuda.empty_cache()
    # del grad, adv_latent_x0, adv_latent_xt, noise_pred, cost 
    
    

    torch.cuda.empty_cache()
    gc.collect()
    
    adv_img = adv_img.detach()
    

    adv_img_cpu = adv_img.cpu().squeeze(0)  # shape: (3, H, W)
    adv_img_cpu = (adv_img_cpu + 1) / 2  # Map to [0, 1]
    adv_img_pil = to_pil_image(adv_img_cpu)

    ## Save as PNG
    if args.mem == "member":
        img_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/SDv1_5_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/adv_img"
    elif args.mem == "non_member":
        img_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/SDv1_5_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/adv_img"
    else:
        ValueError("args.mem was not satisfied.")

    os.makedirs(img_dir, exist_ok=True)
    save_path = os.path.join(img_dir, f"{prompt_id}_anchor{anchor}_OptimIter{optim_iters}_eps{eps}_step{step_size}.png")
    adv_img_pil.save(save_path)
    
    
    
    
    
    #### 2. Text embedding ####
    ## New
    set_random_seed(args.gen_seed)
    
    rnd_text_emb = init_text_embeddings[1].unsqueeze(0).detach() # (1, 77, 768)
    
    with torch.no_grad():
        latent = pipe.vae.encode(adv_img)
        latent = 0.18215 * latent.latent_dist.sample()


        inverted_latents = pipe.inversion(prompt=None, latents=latent, text_embeddings=rnd_text_emb, guidance_scale=1.) ## default: g=1.0





    latent_cur = inverted_latents[anchor+1]
    latent_prev = inverted_latents[anchor]

    timestep = timesteps[anchor+1]
    timestep = torch.tensor([timestep], device=device)
    
    next_timestep = timesteps[anchor]
    next_timestep = torch.tensor([next_timestep], device=device)
    
    # rnd_text_emb.requires_grad_(True)
    rnd_text_emb = rnd_text_emb.detach().clone().requires_grad_(True)
    ### From Null-text inversion ###
    # optimizer = Adam([rnd_text_emb], lr=1e-2 * (1. - t / 100.))
    optimizer = Adam([rnd_text_emb], lr=1e-2)
    
    

    def get_noise_pred_single(pipe, latents, t, context):
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=context).sample
        # noise_pred = pipe.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    with torch.no_grad():
        noise_pred_uncond = get_noise_pred_single(pipe, latent_cur, timestep, uncond_emb)
        
    
    
    pbar = tqdm(range(iters))
    for it in pbar:
        
        
        noise_pred_cond = get_noise_pred_single(pipe, latent_cur, timestep, rnd_text_emb)

        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)

        latents_prev_rec = pipe.scheduler.step(noise_pred, timestep, next_timestep, latent_cur, **extra_step_kwargs).prev_sample
        
    
        loss = (mse_loss(latents_prev_rec, latent_prev)).to(pipe.device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ## Early Stopping ##
        # loss_item = loss.item()
        # if loss_item < epsilon + t * 2e-5:
        #     break
        
        if pbar is not None:
            pbar.set_description(
                f"Image:{prompt_id} | Optimizing: t={timestep.item()}->{next_timestep.item()} | Iter {it} - Current loss: {loss.item():.8f}"
            )
        
    # cond_embeddings_list.append(rnd_text_emb[:1].detach())/
    
    
    rnd_text_emb = rnd_text_emb.detach().cpu()
    
    rnd_text_emb_npy = rnd_text_emb.numpy()    


    ## Save .npy
    if args.mem == "member":
        save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/SDv1_5_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/perturb_emb"
    elif args.mem == "non_member":
        save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/SDv1_5_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/perturb_emb"
    else:
        ValueError("args.mem was not satisfied.")

    os.makedirs(save_dir, exist_ok=True)
    np.save(save_dir+f"/{prompt_id}_anchor{anchor}_iter{iters}_{init}_Adam.npy", rnd_text_emb_npy)
    


 
def main_adv_text_per_img_8(args):
    
    
    
    
    def encode_prompt_(caption, tokenizer, text_encoder):
        captions = [caption]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(text_encoder.device)

        encoder_hidden_states = text_encoder(input_ids)[0]
        
        return encoder_hidden_states
    
    def load_pipeline(ckpt_path, device='cuda:0'):
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe
    
    
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "runwayml/stable-diffusion-v1-5"

    ##############################################################
    # ckpt_path = "/mnt/nas5/joonsung/2025/ckpts/sd-pokemon-checkpoint/sd-pokemon-checkpoint"
    
    # ckpt_path = 'runwayml/stable-diffusion-v1-5'
    

    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.ckpt_path, subfolder="tokenizer", revision=None
    # )
    # # tokenizer = tokenizer.to(device)
    # # tokenizer = tokenizer.cuda()

    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.ckpt_path, subfolder="text_encoder", revision=None
    # )
    # text_encoder = text_encoder.to(device)

    # vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    # vae = vae.to(device)

    # unet = UNet2DConditionModel.from_pretrained(
    #     args.ckpt_path, subfolder="unet", revision=None
    # )
    # unet = unet.to(device)
    
    # text_encoder.requires_grad_(False)
    
    # for p in text_encoder.parameters():
    #     p.requires_grad = False
    
    pipe = load_pipeline(args.ckpt_path, device)
    ##############################################################
    set_random_seed(args.gen_seed)


    

    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    
    image = Image.open(args.img_path).convert("RGB")
    images = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 512, 512)
    
    prompt_id = args.img_path.split('/')[-1].split('.')[0]
    
    


        

    mse_loss = nn.MSELoss()
    
    garbage_prompt = ""
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[1].unsqueeze(0).detach()
    
    
    
    ###################################
    
    
    anchor = args.anchor
    until = args.until
    


    ## blip2
    if args.init == "ori":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/member_captions.jsonl"
        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/non_member_captions.jsonl"
        else:
             ValueError("args.mem was not satisfied.")
             
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["filename"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )
    
        init = args.init
        
    
    ## ori - mem
    elif args.init == "clip_interrogator":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_members_caption_output.jsonl"

        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_non_members_caption_output.jsonl"
            
            
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["prompt_id"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )

        init = args.init

        
        
    else:
        ValueError("args.init was not satisfied.")

    
    # noise_scale = 0.1
    optim_iters = args.OptimIter
    iters = args.iters
    eps = args.eps
    step_size = eps/2.
    GUIDANCE_SCALE = 7.5
    # epsilon = 1e-5
    # step_size > 0.01: loss increase
    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
    
    # for p in pipe.text_encoder.parameters():
    #     p.requires_grad = False
    

    
    
    
    #### 1. Adv Example ####
    images = images*2. - 1.
    
    set_random_seed(args.gen_seed)
    adv_img = images.clone().detach() + (torch.rand(*images.shape)*2*eps-eps).to(device=device, dtype=torch.float32)

    
    
    
    
    ## 1. Uncond. DDIM inversion ##
    # set_random_seed(args.gen_seed)
    # with torch.no_grad():
    #     inverted_latents = invert(pipe, anchor, latent, garbage_prompt, device=device, guidance_scale=0, num_inference_steps=50)
        
    ## 2. add_noise ##
    # at the below FOR loop
    num_inference_steps = 100 ## SecMI setting

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    # print(pipe.scheduler.timesteps)
    
    timesteps = list(range(0, 1000, 10))
    
    # trg_noise = torch.randn(inverted_latent.shape).to(device=device, dtype=torch.float32)
    

    
    # for p in pipe.unet.parameters():
    #     print(p.requires_grad) ## all True
        
    
    pbar_adv = tqdm(range(optim_iters))

    ## **** ##
    # npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_1.npy"
    # rnd_noise_np = np.load(npy_path)
    # rnd_noise = torch.from_numpy(rnd_noise_np).to(device)
    
    for j in range(anchor, until, -1):
        set_random_seed(args.gen_seed)
        # adv_img.requires_grad_(True)
        
        
        # optimizer = Adam([adv_img], lr=0.001) # 0.001
        # # optimizer = SGD([rnd_text_emb], lr=0.1) ## xxx 
        
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                             lr_lambda=lambda epoch: 0.95 ** epoch,
        #                             last_epoch=-1,
        #                             verbose=False)


        
        ## -------------------- old -------------------- ##
        # t=j
        # timestep = timesteps[t]
        # timestep = torch.tensor([timestep], device=device)
        
        ## -------------------- old -------------------- ##
        
        ## anchor == 20
        
        for it in pbar_adv:

            adv_img = adv_img.detach().clone().requires_grad_(True)
            
            
            ## -------------------- NEW -------------------- ##
            rnd = args.adv_rnd
            t = torch.randint(j-(rnd//2), j + (rnd//2+1), (1,)).item()  # 0 ~ t 사이의 랜덤 정수
            
            ## anchor = 130 -> {110, 120, 130, 140, 150}

            # timesteps에서 해당 값 추출
            timestep = timesteps[t]
            timestep = torch.tensor([timestep], device=device)

            ## -------------------- NEW -------------------- ##  
            
            ## 2. ##
            pipe.unet.zero_grad()
            pipe.vae.zero_grad()
            
            actual_step_size = step_size - (step_size - step_size / 100) / optim_iters * it
            # adv_latent_x0 = encode_image_grad(pipe, adv_img, dtype=torch.float32)
            
            adv_latent_x0 = pipe.vae.encode(adv_img.to(dtype=torch.float32))
            adv_latent_x0 = 0.18215 * adv_latent_x0.latent_dist.sample()

            
            ## **** ##
            rnd_noise = torch.randn(adv_latent_x0.shape).to(device=device, dtype=adv_latent_x0.dtype)
            
            adv_latent_xt = pipe.scheduler.add_noise(adv_latent_x0.to(device), rnd_noise.to(device), timestep)


            _, _, noise_pred_uncond, noise_pred_text = pipe.mtcnp_adv(perturb_embeds=None, perturb_latent=adv_latent_xt, prompt=pred_prompt, anchor=t, guidance_scale=7.5) ## default: 7.5


            

            pipe.unet.zero_grad()
             
            ## 1. Memorization ##
            if args.type == "Memorized":
                cost = mse_loss(noise_pred_uncond, noise_pred_text)
                grad, = torch.autograd.grad(cost, [adv_img])
                adv_img = adv_img + grad.sign() * actual_step_size
            
            ## 2. Uncond ~ rnd noise ##
            elif args.type == "Uncond":
                cost = mse_loss(noise_pred_uncond, rnd_noise)
                grad, = torch.autograd.grad(cost, [adv_img])
                adv_img = adv_img - grad.sign() * actual_step_size
            
            
            adv_img = torch.minimum(torch.maximum(adv_img, adv_img - eps), adv_img + eps)
            adv_img.data = torch.clamp(adv_img, min=-1, max=1)
            adv_img.grad = None
            #### torch.cuda.empty_cache()

            ## 2. ## ==> ldm에서는 터짐
            # cost = (mse_loss(rnd_noise, noise_pred) / (anchor-until)).to(pipe.device)
            # cost.backward()
            # grad = adv_img.grad.detach().sign()
            # adv_img = adv_img - actual_step_size*grad
            # eta = torch.clamp(adv_img.data - images.data, min=-eps, max=eps)
            # adv_img = adv_img.detach()
            # adv_img = torch.clamp(adv_img + eta, min=-1, max=1)
            
            
            

            if pbar_adv is not None:
                pbar_adv.set_description(
                    f"Image: {prompt_id} | timestep {timestep.item()} | Iter {it} | eps {eps} --> Step size: {actual_step_size:.4f} / Current loss: {cost.item():.6f}"
                )


            ## 2. Adam
            # latent = encode_image(pipe, adv_img, dtype=torch.float32)
            # rnd_noise = torch.randn(latent.shape).to(device=device, dtype=latent.dtype)
            # latent_cur = pipe.scheduler.add_noise(latent.to(device), rnd_noise.to(device), timestep.to(device))

            # noise_pred_cond = get_noise_pred_single(pipe, latent_cur, timestep, rnd_text_emb)


        
            

        
    
    # torch.cuda.empty_cache()
    # del grad, adv_latent_x0, adv_latent_xt, noise_pred, cost 
    
    

    torch.cuda.empty_cache()
    gc.collect()
    
    adv_img = adv_img.detach()
    

    adv_img_cpu = adv_img.cpu().squeeze(0)  # shape: (3, H, W)
    adv_img_cpu = (adv_img_cpu + 1) / 2  # Map to [0, 1]
    adv_img_pil = to_pil_image(adv_img_cpu)

    #### Save as PNG
    # if args.mem == "member":
    #     img_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/adv_img"
    # elif args.mem == "non_member":
    #     img_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/adv_img"
    # else:
    #     ValueError("args.mem was not satisfied.")
        
    if args.mem == "member":
        img_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/Test_{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/adv_img"
    elif args.mem == "non_member":
        img_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/Test_{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/adv_img"
    else:
        ValueError("args.mem was not satisfied.")

    os.makedirs(img_dir, exist_ok=True)
    save_path = os.path.join(img_dir, f"{prompt_id}_anchor{anchor}_rnd{rnd}_OptimIter{optim_iters}_eps{eps}_step{step_size}.png")
    adv_img_pil.save(save_path)
    
    
    
    
    
    #### 2. Text embedding ####
    ## New
    set_random_seed(args.gen_seed)
    
    rnd_text_emb = init_text_embeddings[1].unsqueeze(0).detach() # (1, 77, 768)
    
    with torch.no_grad():
        latent = pipe.vae.encode(adv_img)
        latent_z0 = 0.18215 * latent.latent_dist.sample()


    # rnd_text_emb.requires_grad_(True)
    rnd_text_emb = rnd_text_emb.detach().clone().requires_grad_(True)
    ### From Null-text inversion ###
    # optimizer = Adam([rnd_text_emb], lr=1e-2 * (1. - t / 100.))
    optimizer = Adam([rnd_text_emb], lr=1e-2)
    
    

    def get_noise_pred_single(pipe, latents, t, context):
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=context).sample
        # noise_pred = pipe.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    
        
    # npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_2.npy"
    # rnd_noise_np = np.load(npy_path)
    # rnd_noise_2 = torch.from_numpy(rnd_noise_np).to(device)
    
    pbar = tqdm(range(iters))
    for it in pbar:
        
        rnd = args.emb_rnd
        t = torch.randint(j-(rnd//2), j+(rnd//2+1), (1,)).item()  # 0 ~ t 사이의 랜덤 정수
        
        ## anchor = 150 -> {110, 120, 130, 140, 150}

        # timesteps에서 해당 값 추출
        timestep = timesteps[t]
        timestep = torch.tensor([timestep], device=device)
        
        # rnd_noise_1 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
        rnd_noise_2 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
            
        # latent_zt_1 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        latent_zt_2 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_2.to(device), timestep)
        
        # noise_pred_uncond = get_noise_pred_single(pipe, latent_zt_1, timestep, uncond_emb.detach())
        noise_pred_cond = get_noise_pred_single(pipe, latent_zt_2, timestep, rnd_text_emb)

        

        # loss_uncond = F.mse_loss(noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
        loss_cond = F.mse_loss(noise_pred_cond.float(), rnd_noise_2.float().to(device), reduction="mean")
        
    
        # loss = (loss_cond - loss_uncond).to(pipe.device)
        
        loss = loss_cond.to(pipe.device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ## Early Stopping ##
        # loss_item = loss.item()
        # if loss_item < epsilon + t * 2e-5:
        #     break
        
        if pbar is not None:
            pbar.set_description(
                # f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f} (cond: {loss_cond.item():.8f}, uncond: {loss_uncond.item():.8f})"
                f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f}"
            )
        
    # cond_embeddings_list.append(rnd_text_emb[:1].detach())/
    
    
    rnd_text_emb = rnd_text_emb.detach().cpu()
    
    rnd_text_emb_npy = rnd_text_emb.numpy()    


    ## Save .npy
    # if args.mem == "member":
    #     save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/perturb_emb"
    # elif args.mem == "non_member":
    #     save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/perturb_emb"
    # else:
    #     ValueError("args.mem was not satisfied.")

    # os.makedirs(save_dir, exist_ok=True)
    
    
    if args.mem == "member":
        save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/Test_{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/perturb_emb"
    elif args.mem == "non_member":
        save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/Test_{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/perturb_emb"
    else:
        ValueError("args.mem was not satisfied.")

    os.makedirs(save_dir, exist_ok=True)
    np.save(save_dir+f"/{prompt_id}_anchor{anchor}_rnd{rnd}_iter{iters}_{init}_Adam.npy", rnd_text_emb_npy)
    




 
def main_adv_text_per_img_9(args):
    
    
    
    
    def encode_prompt_(caption, tokenizer, text_encoder):
        captions = [caption]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(text_encoder.device)

        encoder_hidden_states = text_encoder(input_ids)[0]
        
        return encoder_hidden_states
    
    def load_pipeline(ckpt_path, device='cuda:0'):
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe
    
    
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "runwayml/stable-diffusion-v1-5"

    ##############################################################
    # ckpt_path = "/mnt/nas5/joonsung/2025/ckpts/sd-pokemon-checkpoint/sd-pokemon-checkpoint"
    
    # ckpt_path = 'runwayml/stable-diffusion-v1-5'
    

    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.ckpt_path, subfolder="tokenizer", revision=None
    # )
    # # tokenizer = tokenizer.to(device)
    # # tokenizer = tokenizer.cuda()

    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.ckpt_path, subfolder="text_encoder", revision=None
    # )
    # text_encoder = text_encoder.to(device)

    # vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    # vae = vae.to(device)

    # unet = UNet2DConditionModel.from_pretrained(
    #     args.ckpt_path, subfolder="unet", revision=None
    # )
    # unet = unet.to(device)
    
    # text_encoder.requires_grad_(False)
    
    # for p in text_encoder.parameters():
    #     p.requires_grad = False
    
    pipe = load_pipeline(args.ckpt_path, device)
    ##############################################################
    set_random_seed(args.gen_seed)


    

    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    
    image = Image.open(args.img_path).convert("RGB")
    images = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 512, 512)
    
    prompt_id = args.img_path.split('/')[-1].split('.')[0]
    
    


        

    mse_loss = nn.MSELoss()
    
    garbage_prompt = ""
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[1].unsqueeze(0).detach()
    
    
    
    ###################################
    
    
    anchor = args.anchor
    until = args.until
    


    ## blip2
    if args.init == "ori":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/member_captions.jsonl"
        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/non_member_captions.jsonl"
        else:
             ValueError("args.mem was not satisfied.")
             
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["filename"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )
    
        init = args.init
        
    
    ## ori - mem
    elif args.init == "clip_interrogator":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_members_caption_output.jsonl"

        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_non_members_caption_output.jsonl"
            
            
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["prompt_id"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )

        init = args.init

        
        
    else:
        ValueError("args.init was not satisfied.")

    
    # noise_scale = 0.1
    optim_iters = args.OptimIter
    iters = args.iters
    eps = args.eps
    step_size = eps/2.
    GUIDANCE_SCALE = 7.5
    # epsilon = 1e-5
    # step_size > 0.01: loss increase
    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
    
    # for p in pipe.text_encoder.parameters():
    #     p.requires_grad = False
    

    
    
    
    #### 1. Adv Example ####
    images = images*2. - 1.
    
    set_random_seed(args.gen_seed)
    adv_img = images.clone().detach() + (torch.rand(*images.shape)*2*eps-eps).to(device=device, dtype=torch.float32)


    
    
    ## 1. Uncond. DDIM inversion ##
    # set_random_seed(args.gen_seed)
    # with torch.no_grad():
    #     inverted_latents = invert(pipe, anchor, latent, garbage_prompt, device=device, guidance_scale=0, num_inference_steps=50)
        
    ## 2. add_noise ##

    ## _________________________________________________________ ##
    # num_inference_steps = 100 ## SecMI setting

    # pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    ## _________________________________________________________ ##

    
    timesteps = list(range(0, 1000, 10))
    
    # trg_noise = torch.randn(inverted_latent.shape).to(device=device, dtype=torch.float32)
    

    
    # for p in pipe.unet.parameters():
    #     print(p.requires_grad) ## all True
        
    
    pbar_adv = tqdm(range(optim_iters))

    ## **** ##
    npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_1.npy"
    rnd_noise_np = np.load(npy_path)
    rnd_noise_1 = torch.from_numpy(rnd_noise_np).to(device)
    
    for j in range(anchor, until, -1):
        set_random_seed(args.gen_seed)
        # adv_img.requires_grad_(True)
        
        
        # optimizer = Adam([adv_img], lr=0.001) # 0.001
        # # optimizer = SGD([rnd_text_emb], lr=0.1) ## xxx 
        
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                             lr_lambda=lambda epoch: 0.95 ** epoch,
        #                             last_epoch=-1,
        #                             verbose=False)


        
        ## -------------------- old -------------------- ##
        # t=j
        # timestep = timesteps[t]
        # timestep = torch.tensor([timestep], device=device)
        
        ## -------------------- old -------------------- ##
        
        ## anchor == 20
        
        for it in pbar_adv:

            adv_img = adv_img.detach().clone().requires_grad_(True)
            
            
            ## -------------------- NEW -------------------- ##
            rnd = args.adv_rnd
            t = torch.randint(j-(rnd//2), j + (rnd//2+1), (1,)).item()  # 0 ~ t 사이의 랜덤 정수
            
            ## anchor = 130 -> {110, 120, 130, 140, 150}

            # timesteps에서 해당 값 추출
            timestep = timesteps[t]
            timestep = torch.tensor([timestep], device=device)

            ## -------------------- NEW -------------------- ##  
            
            ## 2. ##
            pipe.unet.zero_grad()
            pipe.vae.zero_grad()
            
            actual_step_size = step_size - (step_size - step_size / 100) / optim_iters * it
            # adv_latent_x0 = encode_image_grad(pipe, adv_img, dtype=torch.float32)
            
            adv_latent_x0 = pipe.vae.encode(adv_img.to(dtype=torch.float32))
            adv_latent_x0 = 0.18215 * adv_latent_x0.latent_dist.sample()

            
            ## **** ##
            # rnd_noise = torch.randn(adv_latent_x0.shape).to(device=device, dtype=adv_latent_x0.dtype)
            
            adv_latent_xt = pipe.scheduler.add_noise(adv_latent_x0.to(device), rnd_noise_1.to(device), timestep)


            _, _, noise_pred_uncond, noise_pred_text = pipe.mtcnp_adv(perturb_embeds=None, perturb_latent=adv_latent_xt, prompt=pred_prompt, anchor=t, guidance_scale=7.5) ## default: 7.5


            

            pipe.unet.zero_grad()
             
            ## 1. Memorization ##
            if args.type == "Memorized":
                cost = mse_loss(noise_pred_uncond, noise_pred_text)
                grad, = torch.autograd.grad(cost, [adv_img])
                adv_img = adv_img + grad.sign() * actual_step_size
            
            ## 2. Uncond ~ rnd noise ##
            elif args.type == "Uncond":
                cost = mse_loss(noise_pred_uncond, rnd_noise_1)
                grad, = torch.autograd.grad(cost, [adv_img])
                adv_img = adv_img - grad.sign() * actual_step_size
            
            
            adv_img = torch.minimum(torch.maximum(adv_img, adv_img - eps), adv_img + eps)
            adv_img.data = torch.clamp(adv_img, min=-1, max=1)
            adv_img.grad = None
            #### torch.cuda.empty_cache()

            ## 2. ## ==> ldm에서는 터짐
            # cost = (mse_loss(rnd_noise, noise_pred) / (anchor-until)).to(pipe.device)
            # cost.backward()
            # grad = adv_img.grad.detach().sign()
            # adv_img = adv_img - actual_step_size*grad
            # eta = torch.clamp(adv_img.data - images.data, min=-eps, max=eps)
            # adv_img = adv_img.detach()
            # adv_img = torch.clamp(adv_img + eta, min=-1, max=1)
            
            
            

            if pbar_adv is not None:
                pbar_adv.set_description(
                    f"Image: {prompt_id} | timestep {timestep.item()} | Iter {it} | eps {eps} --> Step size: {actual_step_size:.4f} / Current loss: {cost.item():.6f}"
                )


            ## 2. Adam
            # latent = encode_image(pipe, adv_img, dtype=torch.float32)
            # rnd_noise = torch.randn(latent.shape).to(device=device, dtype=latent.dtype)
            # latent_cur = pipe.scheduler.add_noise(latent.to(device), rnd_noise.to(device), timestep.to(device))

            # noise_pred_cond = get_noise_pred_single(pipe, latent_cur, timestep, rnd_text_emb)


        
            

        
    
    # torch.cuda.empty_cache()
    # del grad, adv_latent_x0, adv_latent_xt, noise_pred, cost 
    
    

    torch.cuda.empty_cache()
    gc.collect()
    
    adv_img = adv_img.detach()
    

    adv_img_cpu = adv_img.cpu().squeeze(0)  # shape: (3, H, W)
    adv_img_cpu = (adv_img_cpu + 1) / 2  # Map to [0, 1]
    adv_img_pil = to_pil_image(adv_img_cpu)

    #### Save as PNG
    # if args.mem == "member":
    #     img_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/adv_img"
    # elif args.mem == "non_member":
    #     img_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/adv_img"
    # else:
    #     ValueError("args.mem was not satisfied.")
        
    # if args.mem == "member":
    #     img_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/getBest3_{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/adv_img"
    # elif args.mem == "non_member":
    #     img_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/getBest3_{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/adv_img"
    # else:
    #     ValueError("args.mem was not satisfied.")

    # os.makedirs(img_dir, exist_ok=True)
    # save_path = os.path.join(img_dir, f"{prompt_id}_anchor{anchor}_rnd{rnd}_OptimIter{optim_iters}_eps{eps}_step{step_size}.png")
    # adv_img_pil.save(save_path)
    
    
    
    
    
    #### 2. Text embedding ####
    ## New
    set_random_seed(args.gen_seed)
    
    rnd_text_emb = init_text_embeddings[1].unsqueeze(0).detach() # (1, 77, 768)
    
    with torch.no_grad():
        latent = pipe.vae.encode(adv_img)
        latent_z0 = 0.18215 * latent.latent_dist.sample()
        
        cln_latent = pipe.vae.encode(images)
        cln_latent_z0 = 0.18215 * cln_latent.latent_dist.sample()


    # rnd_text_emb.requires_grad_(True)
    rnd_text_emb = rnd_text_emb.detach().clone().requires_grad_(True)
    ### From Null-text inversion ###
    # optimizer = Adam([rnd_text_emb], lr=1e-2 * (1. - t / 100.))
    optimizer = Adam([rnd_text_emb], lr=1e-2)
    
    

    def get_noise_pred_single(pipe, latents, t, context):
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=context).sample
        # noise_pred = pipe.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    
        
    # npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_2.npy"
    # rnd_noise_np = np.load(npy_path)
    # rnd_noise_2 = torch.from_numpy(rnd_noise_np).to(device)
    
    
    
    pbar = tqdm(range(iters))
    for it in pbar:
        
        rnd = args.emb_rnd
        t = torch.randint(j-(rnd//2), j+(rnd//2+1), (1,)).item()  # 0 ~ t 사이의 랜덤 정수
        
        ## anchor = 150 -> {110, 120, 130, 140, 150}

        # timesteps에서 해당 값 추출
        timestep = timesteps[t]
        timestep = torch.tensor([timestep], device=device)
        
        # rnd_noise_1 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
        # rnd_noise_2 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
            
        # latent_zt_1 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        latent_zt_2 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        
        
        
        noise_pred_uncond = get_noise_pred_single(pipe, latent_zt_2, timestep, uncond_emb.detach())
        noise_pred_cond = get_noise_pred_single(pipe, latent_zt_2, timestep, rnd_text_emb)

        

        loss_uncond = F.mse_loss(noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
        loss_cond = F.mse_loss(noise_pred_cond.float(), rnd_noise_1.float().to(device), reduction="mean")
        
    
        loss = (loss_cond - loss_uncond).to(pipe.device)
        
        # loss = loss_cond.to(pipe.device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ## Eval ##
        with torch.no_grad():
            cln_latent_zt = pipe.scheduler.add_noise(cln_latent_z0.to(device), rnd_noise_1.to(device), timestep).detach()
            cln_noise_pred_uncond = get_noise_pred_single(pipe, cln_latent_zt, timestep, uncond_emb.detach())
            cln_noise_pred_cond = get_noise_pred_single(pipe, cln_latent_zt, timestep, rnd_text_emb)

            
            cln_loss_uncond = F.mse_loss(cln_noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
            cln_loss_cond = F.mse_loss(cln_noise_pred_cond.float(), rnd_noise_1.float().to(device), reduction="mean").to(device)
            
            eval_loss = (cln_loss_cond - cln_loss_uncond).to(pipe.device)
        
        
        
        
        if pbar is not None:
            pbar.set_description(
                # f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f} (cond: {loss_cond.item():.8f}, uncond: {loss_uncond.item():.8f})"
                f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f} ||| clean: {eval_loss.item():.8f}"
            )
        
    # cond_embeddings_list.append(rnd_text_emb[:1].detach())/
    
    
    rnd_text_emb = rnd_text_emb.detach().cpu()
    
    rnd_text_emb_npy = rnd_text_emb.numpy()    


    ## Save .npy
    # if args.mem == "member":
    #     save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/perturb_emb"
    # elif args.mem == "non_member":
    #     save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/perturb_emb"
    # else:
    #     ValueError("args.mem was not satisfied.")

    # os.makedirs(save_dir, exist_ok=True)
    
    
    if args.mem == "member":
        save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/getBest3_{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/perturb_emb"
    elif args.mem == "non_member":
        save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/getBest3_{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/perturb_emb"
    else:
        ValueError("args.mem was not satisfied.")

    os.makedirs(save_dir, exist_ok=True)
    np.save(save_dir+f"/{prompt_id}_anchor{anchor}_rnd{rnd}_iter{iters}_{init}_Adam.npy", rnd_text_emb_npy)
    




def main_adv_text_per_img_10_OnlyEmb(args):
    
    
    
    
    def encode_prompt_(caption, tokenizer, text_encoder):
        captions = [caption]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(text_encoder.device)

        encoder_hidden_states = text_encoder(input_ids)[0]
        
        return encoder_hidden_states
    
    def load_pipeline(ckpt_path, device='cuda:0'):
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe
    
    
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "runwayml/stable-diffusion-v1-5"

    ##############################################################
    # ckpt_path = "/mnt/nas5/joonsung/2025/ckpts/sd-pokemon-checkpoint/sd-pokemon-checkpoint"
    
    # ckpt_path = 'runwayml/stable-diffusion-v1-5'
    

    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.ckpt_path, subfolder="tokenizer", revision=None
    # )
    # # tokenizer = tokenizer.to(device)
    # # tokenizer = tokenizer.cuda()

    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.ckpt_path, subfolder="text_encoder", revision=None
    # )
    # text_encoder = text_encoder.to(device)

    # vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    # vae = vae.to(device)

    # unet = UNet2DConditionModel.from_pretrained(
    #     args.ckpt_path, subfolder="unet", revision=None
    # )
    # unet = unet.to(device)
    
    # text_encoder.requires_grad_(False)
    
    # for p in text_encoder.parameters():
    #     p.requires_grad = False
    
    pipe = load_pipeline(args.ckpt_path, device)
    ##############################################################
    set_random_seed(args.gen_seed)


    

    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    
    image = Image.open(args.img_path).convert("RGB")
    images = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 512, 512)
    
    prompt_id = args.img_path.split('/')[-1].split('_')[0]
    
    


        

    mse_loss = nn.MSELoss()
    
    garbage_prompt = ""
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[1].unsqueeze(0).detach()
    
    
    
    ###################################
    
    
    anchor = args.anchor
    until = args.until
    


    ## blip2
    if args.init == "ori":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/member_captions.jsonl"
        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/non_member_captions.jsonl"
        else:
             ValueError("args.mem was not satisfied.")
             
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["filename"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )
    
        init = args.init
        
    
    ## ori - mem
    elif args.init == "clip_interrogator":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_members_caption_output.jsonl"

        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_non_members_caption_output.jsonl"
            
            
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["prompt_id"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )

        init = args.init

        
        
    else:
        ValueError("args.init was not satisfied.")

    
    # noise_scale = 0.1
    optim_iters = args.OptimIter
    iters = args.iters
    eps = args.eps
    step_size = eps/2.
    GUIDANCE_SCALE = 7.5
    # epsilon = 1e-5
    # step_size > 0.01: loss increase
    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
    
    # for p in pipe.text_encoder.parameters():
    #     p.requires_grad = False
    

    
    
    
    #### 1. Adv Example ####
    
    
    set_random_seed(args.gen_seed)
    
    # pipe.unet.eval()
    # pipe.vae.eval()
    
    adv_img = images*2. - 1.
    
    
    
    timesteps = list(range(0, 1000, 10))
    
    # trg_noise = torch.randn(inverted_latent.shape).to(device=device, dtype=torch.float32)
    

    
    # for p in pipe.unet.parameters():
    #     print(p.requires_grad) ## all True
        
    
    # pbar_adv = tqdm(range(optim_iters))

    # ## **** ##
    npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_1.npy"
    rnd_noise_np = np.load(npy_path)
    rnd_noise_1 = torch.from_numpy(rnd_noise_np).to(device=device, dtype=adv_img.dtype)
    

    
    
    
    #### 2. Text embedding ####
    ## New
    set_random_seed(args.gen_seed)
    
    rnd_text_emb = init_text_embeddings[1].unsqueeze(0).detach() # (1, 77, 768)
    
    with torch.no_grad():
        latent = pipe.vae.encode(adv_img)
        latent_z0 = 0.18215 * latent.latent_dist.sample()
        


    # rnd_text_emb.requires_grad_(True)
    rnd_text_emb = rnd_text_emb.detach().clone().requires_grad_(True)
    ### From Null-text inversion ###
    # optimizer = Adam([rnd_text_emb], lr=1e-2 * (1. - t / 100.))
    optimizer = Adam([rnd_text_emb], lr=args.lr)
    
    

    def get_noise_pred_single(pipe, latents, t, context):
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=context).sample
        # noise_pred = pipe.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    
        
    # adv_dir = "/mnt/nas5/joonsung/2025/adv_ex_emb_LAION/ver10/EMB_Uncond_anchor14_init_blip2_OptimIter1000_iters1000_eps0.3"
    
    # if args.mem == "member":
    #     pre_dir = f"{adv_dir}/members"
    #     save_it_dir = pre_dir + f"/perturb_emb_early_stopped"
    #     os.makedirs(save_it_dir, exist_ok=True)
        
    # elif args.mem == "non_member":
    #     pre_dir = f"{adv_dir}/non_members"
    #     save_it_dir = pre_dir + f"/perturb_emb_early_stopped"
    #     os.makedirs(save_it_dir, exist_ok=True)
    # else:
    #     ValueError("args.mem was not satisfied.")
    
    loss_values = []
    
    pbar = tqdm(range(iters))
    for it in pbar:
        
        rnd = args.emb_rnd
        t = anchor  # 0 ~ t 사이의 랜덤 정수
        
        ## anchor = 150 -> {110, 120, 130, 140, 150}

        # timesteps에서 해당 값 추출
        timestep = timesteps[t]
        timestep = torch.tensor([timestep], device=device)
        
        # rnd_noise_1 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
        # rnd_noise_2 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
            
        # latent_zt_1 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        latent_zt_2 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        
        
        
        # noise_pred_uncond = get_noise_pred_single(pipe, latent_zt_2, timestep, uncond_emb.detach())
        noise_pred_cond = get_noise_pred_single(pipe, latent_zt_2, timestep, rnd_text_emb)

        

        # loss_uncond = F.mse_loss(noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
        loss_cond = F.mse_loss(noise_pred_cond.float(), rnd_noise_1.float().to(device), reduction="mean")
        
    
        # loss = (loss_cond - loss_uncond).to(pipe.device)
        loss = loss_cond.to(pipe.device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        
        if pbar is not None:
            pbar.set_description(
                # f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f} (cond: {loss_cond.item():.8f}, uncond: {loss_uncond.item():.8f})"
                f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f}"
            )
            
        if (it+1) % 100 == 0:
            loss_values.append(loss.item())
            # print(f"loss at iter={it+1}: {loss.item():.5f}")
            print([f"{loss:.6f}" for loss in loss_values])
            

            
        # if (it+1) > 15000 and loss < 0.0001:
        #     rnd_text_emb_npy = rnd_text_emb.detach().cpu().numpy()
        #     save_it_dir = pre_dir + f"/perturb_emb_early_stopped"
        #     np.save(save_it_dir+f"/{prompt_id}_anchor{anchor}_iter{it+1}_loss{loss:.5f}_rnd{rnd}_{init}_Adam.npy", rnd_text_emb_npy)
        #     break
        
        # if (it+1) == iters: 
        #     rnd_text_emb_npy = rnd_text_emb.detach().cpu().numpy()
        #     save_it_dir = pre_dir + f"/perturb_emb_early_stopped"
        #     np.save(save_it_dir+f"/{prompt_id}_anchor{anchor}_iter{it+1}_loss{loss:.5f}_rnd{rnd}_{init}_Adam.npy", rnd_text_emb_npy)
        




def main_adv_text_per_img_10_OnlyEmb_optimizer(args):
    
    
    
    
    def encode_prompt_(caption, tokenizer, text_encoder):
        captions = [caption]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(text_encoder.device)

        encoder_hidden_states = text_encoder(input_ids)[0]
        
        return encoder_hidden_states
    
    def load_pipeline(ckpt_path, device='cuda:0'):
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe
    
    
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "runwayml/stable-diffusion-v1-5"

    ##############################################################
    # ckpt_path = "/mnt/nas5/joonsung/2025/ckpts/sd-pokemon-checkpoint/sd-pokemon-checkpoint"
    
    # ckpt_path = 'runwayml/stable-diffusion-v1-5'
    

    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.ckpt_path, subfolder="tokenizer", revision=None
    # )
    # # tokenizer = tokenizer.to(device)
    # # tokenizer = tokenizer.cuda()

    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.ckpt_path, subfolder="text_encoder", revision=None
    # )
    # text_encoder = text_encoder.to(device)

    # vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    # vae = vae.to(device)

    # unet = UNet2DConditionModel.from_pretrained(
    #     args.ckpt_path, subfolder="unet", revision=None
    # )
    # unet = unet.to(device)
    
    # text_encoder.requires_grad_(False)
    
    # for p in text_encoder.parameters():
    #     p.requires_grad = False
    
    pipe = load_pipeline(args.ckpt_path, device)
    ##############################################################
    set_random_seed(args.gen_seed)


    

    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    
    image = Image.open(args.img_path).convert("RGB")
    images = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 512, 512)
    
    prompt_id = args.img_path.split('/')[-1].split('_')[0]
    
    


        

    mse_loss = nn.MSELoss()
    
    garbage_prompt = ""
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[1].unsqueeze(0).detach()
    
    
    
    ###################################
    
    
    anchor = args.anchor
    until = args.until
    


    ## blip2
    if args.init == "ori":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/member_captions.jsonl"
        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/non_member_captions.jsonl"
        else:
             ValueError("args.mem was not satisfied.")
             
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["filename"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )
    
        init = args.init
        
    
    ## ori - mem
    elif args.init == "clip_interrogator":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_members_caption_output.jsonl"

        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_non_members_caption_output.jsonl"
            
            
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["prompt_id"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )

        init = args.init

        
        
    else:
        ValueError("args.init was not satisfied.")

    
    # noise_scale = 0.1
    optim_iters = args.OptimIter
    iters = args.iters
    eps = args.eps
    step_size = eps/2.
    GUIDANCE_SCALE = 7.5
    # epsilon = 1e-5
    # step_size > 0.01: loss increase
    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
    
    # for p in pipe.text_encoder.parameters():
    #     p.requires_grad = False
    

    
    
    
    #### 1. Adv Example ####
    
    
    set_random_seed(args.gen_seed)
    
    # pipe.unet.eval()
    # pipe.vae.eval()
    
    adv_img = images*2. - 1.
    
    
    
    timesteps = list(range(0, 1000, 10))
    
    # trg_noise = torch.randn(inverted_latent.shape).to(device=device, dtype=torch.float32)
    

    
    # for p in pipe.unet.parameters():
    #     print(p.requires_grad) ## all True
        
    
    # pbar_adv = tqdm(range(optim_iters))

    # ## **** ##
    npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_1.npy"
    rnd_noise_np = np.load(npy_path)
    rnd_noise_1 = torch.from_numpy(rnd_noise_np).to(device=device, dtype=adv_img.dtype)
    

    
    
    
    #### 2. Text embedding ####
    ## New
    set_random_seed(args.gen_seed)
    
    rnd_text_emb = init_text_embeddings[1].unsqueeze(0).detach() # (1, 77, 768)
    
    with torch.no_grad():
        latent = pipe.vae.encode(adv_img)
        latent_z0 = 0.18215 * latent.latent_dist.sample()
        


    # rnd_text_emb.requires_grad_(True)
    rnd_text_emb = rnd_text_emb.detach().clone().requires_grad_(True)
    
    ### Optimization ###
    optimizer = Adam([rnd_text_emb], lr=args.lr)
    # optimizer = SGD([rnd_text_emb], lr=args.lr, momentum=0.9)
    # optimizer = RMSprop([rnd_text_emb], lr=args.lr, alpha=0.99, eps=1e-8, momentum=0.9)    
    # optimizer = Lion([rnd_text_emb], lr=args.lr)
    

    def get_noise_pred_single(pipe, latents, t, context):
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=context).sample
        # noise_pred = pipe.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    
        
    # adv_dir = "/mnt/nas5/joonsung/2025/adv_ex_emb_LAION/ver10/EMB_Uncond_anchor14_init_blip2_OptimIter1000_iters1000_eps0.3"
    
    # if args.mem == "member":
    #     pre_dir = f"{adv_dir}/members"
    #     save_it_dir = pre_dir + f"/perturb_emb_early_stopped"
    #     os.makedirs(save_it_dir, exist_ok=True)
        
    # elif args.mem == "non_member":
    #     pre_dir = f"{adv_dir}/non_members"
    #     save_it_dir = pre_dir + f"/perturb_emb_early_stopped"
    #     os.makedirs(save_it_dir, exist_ok=True)
    # else:
    #     ValueError("args.mem was not satisfied.")
    
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    starter.record()   # 🔥 GPU timer start

    loss_values = []
    
    pbar = tqdm(range(iters))
    for it in pbar:
        
        rnd = args.emb_rnd
        t = anchor  # 0 ~ t 사이의 랜덤 정수
        
        ## anchor = 150 -> {110, 120, 130, 140, 150}

        # timesteps에서 해당 값 추출
        timestep = timesteps[t]
        timestep = torch.tensor([timestep], device=device)
        
        # rnd_noise_1 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
        # rnd_noise_2 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
            
        # latent_zt_1 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        latent_zt_2 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        
        
        
        # noise_pred_uncond = get_noise_pred_single(pipe, latent_zt_2, timestep, uncond_emb.detach())
        noise_pred_cond = get_noise_pred_single(pipe, latent_zt_2, timestep, rnd_text_emb)

        

        # loss_uncond = F.mse_loss(noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
        loss_cond = F.mse_loss(noise_pred_cond.float(), rnd_noise_1.float().to(device), reduction="mean")
        
    
        # loss = (loss_cond - loss_uncond).to(pipe.device)
        loss = loss_cond.to(pipe.device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        peak = torch.cuda.max_memory_allocated() / 1024**2
        # print("Peak memory used:", peak, "MB")
        
        if pbar is not None:
            pbar.set_description(
                # f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f} (cond: {loss_cond.item():.8f}, uncond: {loss_uncond.item():.8f})"
                f"Image:{prompt_id} | VRAM: {peak}MB | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f}"
            )
            
        # if (it+1) % 100 == 0:
        #     loss_values.append(loss.item())
        #     # print(f"loss at iter={it+1}: {loss.item():.5f}")
        #     print([f"{loss:.6f}" for loss in loss_values])
            

            
        # if (it+1) > 15000 and loss < 0.0001:
        #     rnd_text_emb_npy = rnd_text_emb.detach().cpu().numpy()
        #     save_it_dir = pre_dir + f"/perturb_emb_early_stopped"
        #     np.save(save_it_dir+f"/{prompt_id}_anchor{anchor}_iter{it+1}_loss{loss:.5f}_rnd{rnd}_{init}_Adam.npy", rnd_text_emb_npy)
        #     break
        
        # if (it+1) == iters: 
        #     rnd_text_emb_npy = rnd_text_emb.detach().cpu().numpy()
        #     save_it_dir = pre_dir + f"/perturb_emb_early_stopped"
        #     np.save(save_it_dir+f"/{prompt_id}_anchor{anchor}_iter{it+1}_loss{loss:.5f}_rnd{rnd}_{init}_Adam.npy", rnd_text_emb_npy)
        
    torch.cuda.synchronize()
    ender.record()  # 🔥 GPU timer end
    torch.cuda.synchronize()

    # Total GPU time (ms → seconds)
    gpu_time_ms = starter.elapsed_time(ender)
    gpu_time_sec = gpu_time_ms / 1000.0

    print(f"\n🔥 Total GPU time: {gpu_time_sec:.4f} seconds")
    print(f"🔥 Average per iteration: {gpu_time_sec/iters:.6f} sec/iter\n")
            




def main_adv_text_per_img_10_fromEmb(args):
    
    
    
    
    def encode_prompt_(caption, tokenizer, text_encoder):
        captions = [caption]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(text_encoder.device)

        encoder_hidden_states = text_encoder(input_ids)[0]
        
        return encoder_hidden_states
    
    def load_pipeline(ckpt_path, device='cuda:0'):
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe
    
    
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "runwayml/stable-diffusion-v1-5"

    ##############################################################
    # ckpt_path = "/mnt/nas5/joonsung/2025/ckpts/sd-pokemon-checkpoint/sd-pokemon-checkpoint"
    
    # ckpt_path = 'runwayml/stable-diffusion-v1-5'
    

    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.ckpt_path, subfolder="tokenizer", revision=None
    # )
    # # tokenizer = tokenizer.to(device)
    # # tokenizer = tokenizer.cuda()

    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.ckpt_path, subfolder="text_encoder", revision=None
    # )
    # text_encoder = text_encoder.to(device)

    # vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    # vae = vae.to(device)

    # unet = UNet2DConditionModel.from_pretrained(
    #     args.ckpt_path, subfolder="unet", revision=None
    # )
    # unet = unet.to(device)
    
    # text_encoder.requires_grad_(False)
    
    # for p in text_encoder.parameters():
    #     p.requires_grad = False
    
    pipe = load_pipeline(args.ckpt_path, device)
    ##############################################################
    set_random_seed(args.gen_seed)


    

    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    
    image = Image.open(args.img_path).convert("RGB")
    images = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 512, 512)
    
    prompt_id = args.img_path.split('/')[-1].split('_')[0]
    
    


        

    mse_loss = nn.MSELoss()
    
    garbage_prompt = ""
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[1].unsqueeze(0).detach()
    
    
    
    ###################################
    
    
    anchor = args.anchor
    until = args.until
    


    ## blip2
    if args.init == "ori":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/member_captions.jsonl"
        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/non_member_captions.jsonl"
        else:
             ValueError("args.mem was not satisfied.")
             
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["filename"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )
    
        init = args.init
        
    
    ## ori - mem
    elif args.init == "clip_interrogator":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_members_caption_output.jsonl"

        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_non_members_caption_output.jsonl"
            
            
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["prompt_id"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )

        init = args.init

        
        
    else:
        ValueError("args.init was not satisfied.")

    
    # noise_scale = 0.1
    optim_iters = args.OptimIter
    iters = args.iters
    eps = args.eps
    step_size = eps/2.
    GUIDANCE_SCALE = 7.5
    # epsilon = 1e-5
    # step_size > 0.01: loss increase
    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
    
    # for p in pipe.text_encoder.parameters():
    #     p.requires_grad = False
    

    
    
    
    #### 1. Adv Example ####
    
    
    set_random_seed(args.gen_seed)
    
    # pipe.unet.eval()
    # pipe.vae.eval()
    
    adv_img = images*2. - 1.
    # print(torch.min(adv_img)) ## -1
    # print(torch.max(adv_img)) ## 1
    
    
    
    timesteps = list(range(0, 1000, 10))
    
    # trg_noise = torch.randn(inverted_latent.shape).to(device=device, dtype=torch.float32)
    

    
    # for p in pipe.unet.parameters():
    #     print(p.requires_grad) ## all True
        
    
    # pbar_adv = tqdm(range(optim_iters))

    # ## **** ##
    npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_1.npy"
    rnd_noise_np = np.load(npy_path)
    rnd_noise_1 = torch.from_numpy(rnd_noise_np).to(device=device, dtype=adv_img.dtype)
    

    
    
    
    #### 2. Text embedding ####
    ## New
    set_random_seed(args.gen_seed)
    
    # rnd_text_emb = init_text_embeddings[1].unsqueeze(0).detach() # (1, 77, 768)
    
    ## ============== LOAD .npy ============== ##
    adv_dir = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver10/Uncond_anchor14_init_clip_interrogator_OptimIter1000_iters1000_eps0.3"
    
    if args.mem == "member":
        folder_path = adv_dir+"/members/perturb_emb_iter1000_lr0.06"
    elif args.mem == "non_member":
        folder_path = adv_dir+"/non_members/perturb_emb_iter1000_lr0.06"
    else:
        ValueError("args.mem was not satisfied.")
    
    npy_files = sorted([
        f for f in os.listdir(folder_path)
        if f.startswith(prompt_id) and f.endswith(".npy")
    ])
    
    if len(npy_files) != 1:
        ValueError("WRONG .npy!")
    else:
        print("Loaded .npy...")
    
    
    rnd_text_emb = np.load(os.path.join(folder_path, npy_files[0]))
    rnd_text_emb = torch.from_numpy(rnd_text_emb).float().to(device)
    
    ## ========================================== ##
    
    with torch.no_grad():
        latent = pipe.vae.encode(adv_img)
        latent_z0 = 0.18215 * latent.latent_dist.sample()
        


    # rnd_text_emb.requires_grad_(True)
    rnd_text_emb = rnd_text_emb.detach().clone().requires_grad_(True)
    ### From Null-text inversion ###
    # optimizer = Adam([rnd_text_emb], lr=1e-2 * (1. - t / 100.))
    optimizer = Adam([rnd_text_emb], lr=args.lr)
    
    

    def get_noise_pred_single(pipe, latents, t, context):
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=context).sample
        # noise_pred = pipe.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    
    
    
    ## ============== Make Folder ============== ##
    iter_list = list(range(500, args.iters+1, 500))
    
    
    if args.mem == "member":
        pre_dir = f"{adv_dir}/members"
    elif args.mem == "non_member":
        pre_dir = f"{adv_dir}/non_members"
    else:
        ValueError("args.mem was not satisfied.")
        
    for j in iter_list:
        save_it_dir = pre_dir + f"/perturb_emb_iter{1000+j}_lr{args.lr}"
        os.makedirs(save_it_dir, exist_ok=True)
        
    ## ========================================== ##

    loss_values = []
    
    pbar = tqdm(range(iters))
    for it in pbar:
        
        rnd = args.emb_rnd
        t = anchor  # 0 ~ t 사이의 랜덤 정수
        
        ## anchor = 150 -> {110, 120, 130, 140, 150}

        # timesteps에서 해당 값 추출
        timestep = timesteps[t]
        timestep = torch.tensor([timestep], device=device)
        
        # rnd_noise_1 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
        # rnd_noise_2 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
            
        # latent_zt_1 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        latent_zt_2 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        
        
        
        # noise_pred_uncond = get_noise_pred_single(pipe, latent_zt_2, timestep, uncond_emb.detach())
        noise_pred_cond = get_noise_pred_single(pipe, latent_zt_2, timestep, rnd_text_emb)

        

        # loss_uncond = F.mse_loss(noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
        loss_cond = F.mse_loss(noise_pred_cond.float(), rnd_noise_1.float().to(device), reduction="mean")
        
    
        # loss = (loss_cond - loss_uncond).to(pipe.device)
        loss = loss_cond.to(pipe.device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        
        if pbar is not None:
            pbar.set_description(
                # f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f} (cond: {loss_cond.item():.8f}, uncond: {loss_uncond.item():.8f})"
                f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f}"
            )
            
        if (it+1) % 100 == 0:
            loss_values.append(loss.item())
            # print(f"loss at iter={it+1}: {loss.item():.5f}")
            print([f"{loss:.6f}" for loss in loss_values])
            

            
        if (it+1) in iter_list:
            rnd_text_emb_npy = rnd_text_emb.detach().cpu().numpy()
            save_it_dir = pre_dir + f"/perturb_emb_iter{1000+it+1}_lr{args.lr}"
            np.save(save_it_dir+f"/{prompt_id}_anchor{anchor}_iter{1000+it+1}_loss{loss:.6f}_rnd{rnd}_{init}_Adam.npy", rnd_text_emb_npy)

        
        # if (it+1) == iters: 
        #     rnd_text_emb_npy = rnd_text_emb.detach().cpu().numpy()
        #     save_it_dir = pre_dir + f"/perturb_emb_early_stopped"
        #     np.save(save_it_dir+f"/{prompt_id}_anchor{anchor}_iter{it+1}_loss{loss:.5f}_rnd{rnd}_{init}_Adam.npy", rnd_text_emb_npy)
        


## ver 21 ##
def main_rnd_adv_text_10(args):
    
    
    
    
    def encode_prompt_(caption, tokenizer, text_encoder):
        captions = [caption]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(text_encoder.device)

        encoder_hidden_states = text_encoder(input_ids)[0]
        
        return encoder_hidden_states
    
    def load_pipeline(ckpt_path, device='cuda:0'):
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe
    
    
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "runwayml/stable-diffusion-v1-5"

    ##############################################################
    # ckpt_path = "/mnt/nas5/joonsung/2025/ckpts/sd-pokemon-checkpoint/sd-pokemon-checkpoint"
    
    # ckpt_path = 'runwayml/stable-diffusion-v1-5'
    

    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.ckpt_path, subfolder="tokenizer", revision=None
    # )
    # # tokenizer = tokenizer.to(device)
    # # tokenizer = tokenizer.cuda()

    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.ckpt_path, subfolder="text_encoder", revision=None
    # )
    # text_encoder = text_encoder.to(device)

    # vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    # vae = vae.to(device)

    # unet = UNet2DConditionModel.from_pretrained(
    #     args.ckpt_path, subfolder="unet", revision=None
    # )
    # unet = unet.to(device)
    
    # text_encoder.requires_grad_(False)
    
    # for p in text_encoder.parameters():
    #     p.requires_grad = False
    
    pipe = load_pipeline(args.ckpt_path, device)
    ##############################################################
    set_random_seed(args.gen_seed)


    

    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    
    image = Image.open(args.img_path).convert("RGB")
    images = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 512, 512)
    
    prompt_id = args.img_path.split('/')[-1].split('.')[0]
    
    


        

    mse_loss = nn.MSELoss()
    
    garbage_prompt = ""
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[1].unsqueeze(0).detach()
    
    
    
    ###################################
    
    
    anchor = args.anchor
    until = args.until
    


    ## blip2
    if args.init == "ori":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/member_captions.jsonl"
        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/non_member_captions.jsonl"
        else:
             ValueError("args.mem was not satisfied.")
             
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["filename"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )
    
        init = args.init
        
    
    ## ori - mem
    elif args.init == "clip_interrogator":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_members_caption_output.jsonl"

        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_non_members_caption_output.jsonl"
            
            
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["prompt_id"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )

        init = args.init

        
        
    else:
        ValueError("args.init was not satisfied.")

    
    # noise_scale = 0.1
    optim_iters = args.OptimIter
    iters = args.iters
    eps = args.eps
    step_size = eps/2.
    GUIDANCE_SCALE = 7.5
    # epsilon = 1e-5
    # step_size > 0.01: loss increase
    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
    
    # for p in pipe.text_encoder.parameters():
    #     p.requires_grad = False
    

    
    
    
    #### 1. Adv Example ####
    images = images*2. - 1.
    
    images = images.clone().detach() + (torch.rand(*images.shape)*2*eps-eps).to(device=device, dtype=torch.float32)  
    adv_img = torch.clamp(images, -1., 1.)
    
    set_random_seed(args.gen_seed)
    
    
    
    
    #### 2. Text embedding ####
    ## New
    set_random_seed(args.gen_seed)
    
    folder_dir =  f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/clean_img_ver{args.ver}/{args.type}_anchor{anchor}_init_{init}_eps{args.eps}_iters{iters}"

    #### 2. Text embedding ####
    ## New
    set_random_seed(args.gen_seed)
    
    rnd_text_emb = init_text_embeddings[1].unsqueeze(0).detach() # (1, 77, 768)
    
    with torch.no_grad():
        latent = pipe.vae.encode(adv_img)
        latent_z0 = 0.18215 * latent.latent_dist.sample()
        
        cln_latent = pipe.vae.encode(images)
        cln_latent_z0 = 0.18215 * cln_latent.latent_dist.sample()


    # rnd_text_emb.requires_grad_(True)
    rnd_text_emb = rnd_text_emb.detach().clone().requires_grad_(True)
    ### From Null-text inversion ###
    # optimizer = Adam([rnd_text_emb], lr=1e-2 * (1. - t / 100.))
    optimizer = Adam([rnd_text_emb], lr=args.lr)
    
    

    def get_noise_pred_single(pipe, latents, t, context):
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=context).sample
        # noise_pred = pipe.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    
        
    iters_list = list(range(100, iters + 1, 100))
    # print(iters_list)
    
    if args.mem == "member":
        pre_dir = f"{folder_dir}/members"
        for iteration in iters_list:
            save_it_dir = pre_dir + f"/perturb_emb_iter{iteration}_lr{args.lr}"
            os.makedirs(save_it_dir, exist_ok=True)
        
    elif args.mem == "non_member":
        pre_dir = f"{folder_dir}/non_members"
        for iteration in iters_list:
            save_it_dir = pre_dir + f"/perturb_emb_iter{iteration}_lr{args.lr}"
            os.makedirs(save_it_dir, exist_ok=True)
    else:
        ValueError("args.mem was not satisfied.")
    
    
    timesteps = list(range(0, 1000, 10))
    
    npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_1.npy"
    rnd_noise_np = np.load(npy_path)
    rnd_noise_1 = torch.from_numpy(rnd_noise_np).to(device=device, dtype=adv_img.dtype)
    
    pbar = tqdm(range(iters))
    for it in pbar:
        
        t = anchor
        
        ## anchor = 150 -> {110, 120, 130, 140, 150}

        # timesteps에서 해당 값 추출
        timestep = timesteps[t]
        timestep = torch.tensor([timestep], device=device)
        
        # rnd_noise_1 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
        # rnd_noise_2 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
            
        # latent_zt_1 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        latent_zt_2 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        
        
        
        # noise_pred_uncond = get_noise_pred_single(pipe, latent_zt_2, timestep, uncond_emb.detach())
        noise_pred_cond = get_noise_pred_single(pipe, latent_zt_2, timestep, rnd_text_emb)

        

        # loss_uncond = F.mse_loss(noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
        loss_cond = F.mse_loss(noise_pred_cond.float(), rnd_noise_1.float().to(device), reduction="mean")
        
    
        # loss = (loss_cond - loss_uncond).to(pipe.device)
        loss = loss_cond.to(pipe.device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ## Eval ##
        with torch.no_grad():
            cln_latent_zt = pipe.scheduler.add_noise(cln_latent_z0.to(device), rnd_noise_1.to(device), timestep).detach()
            cln_noise_pred_uncond = get_noise_pred_single(pipe, cln_latent_zt, timestep, uncond_emb.detach())
            cln_noise_pred_cond = get_noise_pred_single(pipe, cln_latent_zt, timestep, rnd_text_emb)

            
            cln_loss_uncond = F.mse_loss(cln_noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
            cln_loss_cond = F.mse_loss(cln_noise_pred_cond.float(), rnd_noise_1.float().to(device), reduction="mean").to(device)
            
            eval_loss = (cln_loss_cond - cln_loss_uncond).to(pipe.device)
        
        
        
        
        if pbar is not None:
            pbar.set_description(
                # f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f} (cond: {loss_cond.item():.8f}, uncond: {loss_uncond.item():.8f})"
                f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f} ||| clean: {eval_loss.item():.8f}"
            )
        
        if (it+1) in iters_list:
            rnd_text_emb_npy = rnd_text_emb.detach().cpu().numpy()
            save_it_dir = pre_dir + f"/perturb_emb_iter{it+1}_lr{args.lr}"
            np.save(save_it_dir+f"/{prompt_id}_anchor{anchor}_iter{iters}_{init}_Adam.npy", rnd_text_emb_npy)
            
            
            
            
## ver 20 ##
def main_clean_adv_text_10(args):
    
    
    
    
    def encode_prompt_(caption, tokenizer, text_encoder):
        captions = [caption]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(text_encoder.device)

        encoder_hidden_states = text_encoder(input_ids)[0]
        
        return encoder_hidden_states
    
    def load_pipeline(ckpt_path, device='cuda:0'):
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe
    
    
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "runwayml/stable-diffusion-v1-5"

    ##############################################################
    # ckpt_path = "/mnt/nas5/joonsung/2025/ckpts/sd-pokemon-checkpoint/sd-pokemon-checkpoint"
    
    # ckpt_path = 'runwayml/stable-diffusion-v1-5'
    

    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.ckpt_path, subfolder="tokenizer", revision=None
    # )
    # # tokenizer = tokenizer.to(device)
    # # tokenizer = tokenizer.cuda()

    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.ckpt_path, subfolder="text_encoder", revision=None
    # )
    # text_encoder = text_encoder.to(device)

    # vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    # vae = vae.to(device)

    # unet = UNet2DConditionModel.from_pretrained(
    #     args.ckpt_path, subfolder="unet", revision=None
    # )
    # unet = unet.to(device)
    
    # text_encoder.requires_grad_(False)
    
    # for p in text_encoder.parameters():
    #     p.requires_grad = False
    
    pipe = load_pipeline(args.ckpt_path, device)
    ##############################################################
    set_random_seed(args.gen_seed)


    

    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    
    image = Image.open(args.img_path).convert("RGB")
    images = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 512, 512)
    
    prompt_id = args.img_path.split('/')[-1].split('.')[0]
    
    


        

    mse_loss = nn.MSELoss()
    
    garbage_prompt = ""
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[1].unsqueeze(0).detach()
    
    
    
    ###################################
    
    
    anchor = args.anchor
    until = args.until
    


    ## blip2
    if args.init == "ori":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/member_captions.jsonl"
        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/non_member_captions.jsonl"
        else:
             ValueError("args.mem was not satisfied.")
             
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["filename"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )
    
        init = args.init
        
    
    ## ori - mem
    elif args.init == "clip_interrogator":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_members_caption_output.jsonl"

        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_non_members_caption_output.jsonl"
            
            
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["prompt_id"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )

        init = args.init

        
        
    else:
        ValueError("args.init was not satisfied.")

    
    # noise_scale = 0.1
    optim_iters = args.OptimIter
    iters = args.iters
    eps = args.eps
    step_size = eps/2.
    GUIDANCE_SCALE = 7.5
    # epsilon = 1e-5
    # step_size > 0.01: loss increase
    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
    
    # for p in pipe.text_encoder.parameters():
    #     p.requires_grad = False
    

    
    
    
    #### 1. Adv Example ####
    
    
    set_random_seed(args.gen_seed)
    
    # pipe.unet.eval()
    # pipe.vae.eval()
    
    adv_img = images*2. - 1.
    
    

    ### Save as PNG        
    
    folder_dir =  f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/clean_img_ver{args.ver}/{args.type}_anchor{anchor}_init_{init}_iters{iters}"

    #### 2. Text embedding ####
    ## New
    set_random_seed(args.gen_seed)
    
    rnd_text_emb = init_text_embeddings[1].unsqueeze(0).detach() # (1, 77, 768)
    
    with torch.no_grad():
        latent = pipe.vae.encode(adv_img)
        latent_z0 = 0.18215 * latent.latent_dist.sample()
        
        cln_latent = pipe.vae.encode(images)
        cln_latent_z0 = 0.18215 * cln_latent.latent_dist.sample()


    # rnd_text_emb.requires_grad_(True)
    rnd_text_emb = rnd_text_emb.detach().clone().requires_grad_(True)
    ### From Null-text inversion ###
    # optimizer = Adam([rnd_text_emb], lr=1e-2 * (1. - t / 100.))
    optimizer = Adam([rnd_text_emb], lr=args.lr)
    
    

    def get_noise_pred_single(pipe, latents, t, context):
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=context).sample
        # noise_pred = pipe.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    
        
    iters_list = list(range(100, iters + 1, 100))
    # print(iters_list)
    
    if args.mem == "member":
        pre_dir = f"{folder_dir}/members"
        for iteration in iters_list:
            save_it_dir = pre_dir + f"/perturb_emb_iter{iteration}_lr{args.lr}"
            os.makedirs(save_it_dir, exist_ok=True)
        
    elif args.mem == "non_member":
        pre_dir = f"{folder_dir}/non_members"
        for iteration in iters_list:
            save_it_dir = pre_dir + f"/perturb_emb_iter{iteration}_lr{args.lr}"
            os.makedirs(save_it_dir, exist_ok=True)
    else:
        ValueError("args.mem was not satisfied.")
    
    
    timesteps = list(range(0, 1000, 10))
    
    npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_1.npy"
    rnd_noise_np = np.load(npy_path)
    rnd_noise_1 = torch.from_numpy(rnd_noise_np).to(device=device, dtype=adv_img.dtype)
    
    pbar = tqdm(range(iters))
    for it in pbar:
        
        t = anchor
        
        ## anchor = 150 -> {110, 120, 130, 140, 150}

        # timesteps에서 해당 값 추출
        timestep = timesteps[t]
        timestep = torch.tensor([timestep], device=device)
        
        # rnd_noise_1 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
        # rnd_noise_2 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
            
        # latent_zt_1 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        latent_zt_2 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        
        
        
        # noise_pred_uncond = get_noise_pred_single(pipe, latent_zt_2, timestep, uncond_emb.detach())
        noise_pred_cond = get_noise_pred_single(pipe, latent_zt_2, timestep, rnd_text_emb)

        

        # loss_uncond = F.mse_loss(noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
        loss_cond = F.mse_loss(noise_pred_cond.float(), rnd_noise_1.float().to(device), reduction="mean")
        
    
        # loss = (loss_cond - loss_uncond).to(pipe.device)
        loss = loss_cond.to(pipe.device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ## Eval ##
        with torch.no_grad():
            cln_latent_zt = pipe.scheduler.add_noise(cln_latent_z0.to(device), rnd_noise_1.to(device), timestep).detach()
            cln_noise_pred_uncond = get_noise_pred_single(pipe, cln_latent_zt, timestep, uncond_emb.detach())
            cln_noise_pred_cond = get_noise_pred_single(pipe, cln_latent_zt, timestep, rnd_text_emb)

            
            cln_loss_uncond = F.mse_loss(cln_noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
            cln_loss_cond = F.mse_loss(cln_noise_pred_cond.float(), rnd_noise_1.float().to(device), reduction="mean").to(device)
            
            eval_loss = (cln_loss_cond - cln_loss_uncond).to(pipe.device)
        
        
        
        
        if pbar is not None:
            pbar.set_description(
                # f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f} (cond: {loss_cond.item():.8f}, uncond: {loss_uncond.item():.8f})"
                f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f} ||| clean: {eval_loss.item():.8f}"
            )
        
        if (it+1) in iters_list:
            rnd_text_emb_npy = rnd_text_emb.detach().cpu().numpy()
            save_it_dir = pre_dir + f"/perturb_emb_iter{it+1}_lr{args.lr}"
            np.save(save_it_dir+f"/{prompt_id}_anchor{anchor}_iter{iters}_{init}_Adam.npy", rnd_text_emb_npy)
    
    
    


def main_adv_text_per_img_10(args):
    
    
    
    
    def encode_prompt_(caption, tokenizer, text_encoder):
        captions = [caption]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(text_encoder.device)

        encoder_hidden_states = text_encoder(input_ids)[0]
        
        return encoder_hidden_states
    
    def load_pipeline(ckpt_path, device='cuda:0'):
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe
    
    
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "runwayml/stable-diffusion-v1-5"

    ##############################################################
    # ckpt_path = "/mnt/nas5/joonsung/2025/ckpts/sd-pokemon-checkpoint/sd-pokemon-checkpoint"
    
    # ckpt_path = 'runwayml/stable-diffusion-v1-5'
    

    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.ckpt_path, subfolder="tokenizer", revision=None
    # )
    # # tokenizer = tokenizer.to(device)
    # # tokenizer = tokenizer.cuda()

    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.ckpt_path, subfolder="text_encoder", revision=None
    # )
    # text_encoder = text_encoder.to(device)

    # vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    # vae = vae.to(device)

    # unet = UNet2DConditionModel.from_pretrained(
    #     args.ckpt_path, subfolder="unet", revision=None
    # )
    # unet = unet.to(device)
    
    # text_encoder.requires_grad_(False)
    
    # for p in text_encoder.parameters():
    #     p.requires_grad = False
    
    pipe = load_pipeline(args.ckpt_path, device)
    ##############################################################
    set_random_seed(args.gen_seed)


    

    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    
    image = Image.open(args.img_path).convert("RGB")
    images = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 512, 512)
    
    prompt_id = args.img_path.split('/')[-1].split('.')[0]
    
    


        

    mse_loss = nn.MSELoss()
    
    garbage_prompt = ""
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[1].unsqueeze(0).detach()
    
    
    
    ###################################
    
    
    anchor = args.anchor
    until = args.until
    


    ## blip2
    if args.init == "ori":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/member_captions.jsonl"
        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/non_member_captions.jsonl"
        else:
             ValueError("args.mem was not satisfied.")
             
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["filename"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )
    
        init = args.init
        
    
    ## ori - mem
    elif args.init == "clip_interrogator":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_members_caption_output.jsonl"

        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_non_members_caption_output.jsonl"
            
            
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["prompt_id"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )

        init = args.init

        
        
    else:
        ValueError("args.init was not satisfied.")

    
    # noise_scale = 0.1
    optim_iters = args.OptimIter
    iters = args.iters
    eps = args.eps
    step_size = eps/2.
    GUIDANCE_SCALE = 7.5
    # epsilon = 1e-5
    # step_size > 0.01: loss increase
    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
    
    # for p in pipe.text_encoder.parameters():
    #     p.requires_grad = False
    

    
    
    
    #### 1. Adv Example ####
    
    
    set_random_seed(args.gen_seed)
    
    # pipe.unet.eval()
    # pipe.vae.eval()
    
    images = images*2. - 1.
    
    
    gen = torch.Generator(device=device).manual_seed(args.gen_seed)
    init_noise = (torch.rand(*images.shape, generator=gen, device=device, dtype=torch.float32)*2*eps - eps)
    adv_img = images.clone().detach() + init_noise
    # adv_img = images.clone().detach() + (torch.rand(*images.shape)*2*eps-eps).to(device=device, dtype=torch.float32)

    
    
    
    
    ## 1. Uncond. DDIM inversion ##
    # set_random_seed(args.gen_seed)
    # with torch.no_grad():
    #     inverted_latents = invert(pipe, anchor, latent, garbage_prompt, device=device, guidance_scale=0, num_inference_steps=50)
        
    ## 2. add_noise ##

    ## _________________________________________________________ ##
    # num_inference_steps = 100 ## SecMI setting

    # pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    ## _________________________________________________________ ##

    
    timesteps = list(range(0, 1000, 10))
    
    # trg_noise = torch.randn(inverted_latent.shape).to(device=device, dtype=torch.float32)
    

    
    # for p in pipe.unet.parameters():
    #     print(p.requires_grad) ## all True
        
    
    pbar_adv = tqdm(range(optim_iters))

    ## **** ##
    npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_1.npy"
    rnd_noise_np = np.load(npy_path)
    rnd_noise_1 = torch.from_numpy(rnd_noise_np).to(device=device, dtype=adv_img.dtype)
    
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    starter.record()   # 🔥 GPU timer start
    
    for j in range(anchor, until, -1):
        set_random_seed(args.gen_seed)
        # adv_img.requires_grad_(True)
        
        
        # optimizer = Adam([adv_img], lr=0.001) # 0.001
        # # optimizer = SGD([rnd_text_emb], lr=0.1) ## xxx 
        
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                             lr_lambda=lambda epoch: 0.95 ** epoch,
        #                             last_epoch=-1,
        #                             verbose=False)


        
        ## -------------------- old -------------------- ##
        # t=j
        # timestep = timesteps[t]
        # timestep = torch.tensor([timestep], device=device)
        
        ## -------------------- old -------------------- ##
        
        ## anchor == 20
        
        for it in pbar_adv:

            adv_img = adv_img.detach().clone().requires_grad_(True)
            
            
            ## -------------------- NEW -------------------- ##
            rnd = args.adv_rnd
            t = torch.randint(j-(rnd//2), j + (rnd//2+1), (1,)).item()  # 0 ~ t 사이의 랜덤 정수
            
            ## anchor = 130 -> {110, 120, 130, 140, 150}

            # timesteps에서 해당 값 추출
            timestep = timesteps[t]
            timestep = torch.tensor([timestep], device=device)

            ## -------------------- NEW -------------------- ##  
            
            ## 2. ##
            pipe.unet.zero_grad()
            pipe.vae.zero_grad()
            
            actual_step_size = step_size - (step_size - step_size / 100) / optim_iters * it
            # adv_latent_x0 = encode_image_grad(pipe, adv_img, dtype=torch.float32)
            
            adv_latent_x0 = pipe.vae.encode(adv_img.to(dtype=torch.float32)).latent_dist
            adv_latent_x0 = 0.18215 * adv_latent_x0.mean

            
            ## **** ##
            # rnd_noise = torch.randn(adv_latent_x0.shape).to(device=device, dtype=adv_latent_x0.dtype)
            
            adv_latent_xt = pipe.scheduler.add_noise(adv_latent_x0.to(device), rnd_noise_1.to(device), timestep)


            _, _, noise_pred_uncond, noise_pred_text = pipe.mtcnp_adv(perturb_embeds=None, perturb_latent=adv_latent_xt, prompt=pred_prompt, anchor=t, guidance_scale=7.5) ## default: 7.5



             
            ## 1. Memorization ##
            if args.type == "Memorized":
                cost = mse_loss(noise_pred_uncond, noise_pred_text)
                grad, = torch.autograd.grad(cost, [adv_img])
                adv_img = adv_img + grad.sign() * actual_step_size
            
            ## 2. Uncond ~ rnd noise ##
            elif args.type == "Uncond":
                cost = mse_loss(noise_pred_uncond, rnd_noise_1)
                grad, = torch.autograd.grad(cost, [adv_img])
                adv_img = adv_img - grad.sign() * actual_step_size
            
            
            adv_img = torch.minimum(torch.maximum(adv_img, adv_img - eps), adv_img + eps)
            adv_img.data = torch.clamp(adv_img, min=-1, max=1)
            adv_img.grad = None
            #### torch.cuda.empty_cache()

            ## 2. ## ==> ldm에서는 터짐
            # cost = (mse_loss(rnd_noise, noise_pred) / (anchor-until)).to(pipe.device)
            # cost.backward()
            # grad = adv_img.grad.detach().sign()
            # adv_img = adv_img - actual_step_size*grad
            # eta = torch.clamp(adv_img.data - images.data, min=-eps, max=eps)
            # adv_img = adv_img.detach()
            # adv_img = torch.clamp(adv_img + eta, min=-1, max=1)
            
            # if it+1 == 10:
            #     print("## ****** ##")
            #     print(f"loss at {it+1}: {cost.item():.6f}")
            #     print("## ****** ##")
            
            peak = torch.cuda.max_memory_allocated() / 1024**2

            if pbar_adv is not None:
                pbar_adv.set_description(
                    f"Image: {prompt_id} | timestep {timestep.item()} | Iter {it} | eps {eps} | peak {peak}MB --> Step size: {actual_step_size:.4f} / Current loss: {cost.item():.6f}"
                )


            
            
            ## 2. Adam
            # latent = encode_image(pipe, adv_img, dtype=torch.float32)
            # rnd_noise = torch.randn(latent.shape).to(device=device, dtype=latent.dtype)
            # latent_cur = pipe.scheduler.add_noise(latent.to(device), rnd_noise.to(device), timestep.to(device))

            # noise_pred_cond = get_noise_pred_single(pipe, latent_cur, timestep, rnd_text_emb)

    
    torch.cuda.synchronize()
    ender.record()  # 🔥 GPU timer end
    torch.cuda.synchronize()

    # Total GPU time (ms → seconds)
    gpu_time_ms = starter.elapsed_time(ender)
    gpu_time_sec = gpu_time_ms / 1000.0

    print(f"\n🔥 Total GPU time: {gpu_time_sec:.4f} seconds")
    print(f"🔥 Average per iteration: {gpu_time_sec/iters:.6f} sec/iter\n")
            

        
            

        
    
    # torch.cuda.empty_cache()
    # del grad, adv_latent_x0, adv_latent_xt, noise_pred, cost 
    
    

    torch.cuda.empty_cache()
    gc.collect()
    
    adv_img = adv_img.detach()
    

    # adv_img_cpu = adv_img.cpu().squeeze(0)  # shape: (3, H, W)
    # adv_img_cpu = (adv_img_cpu + 1) / 2  # Map to [0, 1]
    # adv_img_pil = to_pil_image(adv_img_cpu)

    ### Save as PNG
    # folder_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}"
    # if args.mem == "member":
    #     img_dir = f"{folder_dir}/members/adv_img"
    # elif args.mem == "non_member":
    #     img_dir = f"{folder_dir}/non_members/adv_img"
    # else:
    #     ValueError("args.mem was not satisfied.")

    # os.makedirs(img_dir, exist_ok=True)
    # save_path = os.path.join(img_dir, f"{prompt_id}_anchor{anchor}_rnd{rnd}_OptimIter{optim_iters}_eps{eps}_step{step_size}.png")
    # adv_img_pil.save(save_path)
    
    
    
    
    
    #### 2. Text embedding ####
    ## New
    set_random_seed(args.gen_seed)
    
    rnd_text_emb = init_text_embeddings[1].unsqueeze(0).detach() # (1, 77, 768)
    
    with torch.no_grad():
        latent = pipe.vae.encode(adv_img)
        latent_z0 = 0.18215 * latent.latent_dist.sample()
        
        cln_latent = pipe.vae.encode(images)
        cln_latent_z0 = 0.18215 * cln_latent.latent_dist.sample()


    # rnd_text_emb.requires_grad_(True)
    rnd_text_emb = rnd_text_emb.detach().clone().requires_grad_(True)
    ### From Null-text inversion ###
    # optimizer = Adam([rnd_text_emb], lr=1e-2 * (1. - t / 100.))
    optimizer = Adam([rnd_text_emb], lr=args.lr)
    
    

    def get_noise_pred_single(pipe, latents, t, context):
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=context).sample
        # noise_pred = pipe.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    
        
    # npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_2.npy"
    # rnd_noise_np = np.load(npy_path)
    # rnd_noise_2 = torch.from_numpy(rnd_noise_np).to(device)
    
    
    iters_list = list(range(0, iters + 1, 100))
    
    
    # if args.mem == "member":
    #     pre_dir = f"{folder_dir}/members"
    #     for iteration in iters_list:
    #         save_it_dir = pre_dir + f"/perturb_emb_iter{iteration}_lr{args.lr}"
    #         os.makedirs(save_it_dir, exist_ok=True)
        
    # elif args.mem == "non_member":
    #     pre_dir = f"{folder_dir}/non_members"
    #     for iteration in iters_list:
    #         save_it_dir = pre_dir + f"/perturb_emb_iter{iteration}_lr{args.lr}"
    #         os.makedirs(save_it_dir, exist_ok=True)
    # else:
    #     ValueError("args.mem was not satisfied.")
    
    pbar = tqdm(range(iters))
    for it in pbar:
        
        rnd = args.emb_rnd
        t = torch.randint(j-(rnd//2), j+(rnd//2+1), (1,)).item()  # 0 ~ t 사이의 랜덤 정수
        
        ## anchor = 150 -> {110, 120, 130, 140, 150}

        # timesteps에서 해당 값 추출
        timestep = timesteps[t]
        timestep = torch.tensor([timestep], device=device)
        
        # rnd_noise_1 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
        # rnd_noise_2 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
            
        # latent_zt_1 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        latent_zt_2 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        
        
        
        # noise_pred_uncond = get_noise_pred_single(pipe, latent_zt_2, timestep, uncond_emb.detach())
        noise_pred_cond = get_noise_pred_single(pipe, latent_zt_2, timestep, rnd_text_emb)

        

        # loss_uncond = F.mse_loss(noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
        loss_cond = F.mse_loss(noise_pred_cond.float(), rnd_noise_1.float().to(device), reduction="mean")
        
    
        # loss = (loss_cond - loss_uncond).to(pipe.device)
        loss = loss_cond.to(pipe.device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ## Eval ##
        with torch.no_grad():
            cln_latent_zt = pipe.scheduler.add_noise(cln_latent_z0.to(device), rnd_noise_1.to(device), timestep).detach()
            cln_noise_pred_uncond = get_noise_pred_single(pipe, cln_latent_zt, timestep, uncond_emb.detach())
            cln_noise_pred_cond = get_noise_pred_single(pipe, cln_latent_zt, timestep, rnd_text_emb)

            
            cln_loss_uncond = F.mse_loss(cln_noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
            cln_loss_cond = F.mse_loss(cln_noise_pred_cond.float(), rnd_noise_1.float().to(device), reduction="mean").to(device)
            
            eval_loss = (cln_loss_cond - cln_loss_uncond).to(pipe.device)
        
        
        
        
        
        if pbar is not None:
            pbar.set_description(
                # f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f} (cond: {loss_cond.item():.8f}, uncond: {loss_uncond.item():.8f})"
                f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f} ||| clean: {eval_loss.item():.8f}"
            )
        
        if (it+1) in iters_list:
            print(f"loss at iter={it+1}: {loss.item():.5f}")
            
        # if (it+1) in iters_list:
        #     rnd_text_emb_npy = rnd_text_emb.detach().cpu().numpy()
        #     save_it_dir = pre_dir + f"/perturb_emb_iter{it+1}_lr{args.lr}"
        #     np.save(save_it_dir+f"/{prompt_id}_anchor{anchor}_rnd{rnd}_iter{iters}_{init}_Adam.npy", rnd_text_emb_npy)
        
    # cond_embeddings_list.append(rnd_text_emb[:1].detach())/
    
    
    # rnd_text_emb = rnd_text_emb.detach().cpu()
    
    # rnd_text_emb_npy = rnd_text_emb.numpy()    


    ## Save .npy
    # if args.mem == "member":
    #     save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/JS_{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/perturb_emb"
    # elif args.mem == "non_member":
    #     save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/JS_{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/perturb_emb"
    # else:
    #     ValueError("args.mem was not satisfied.")

    # os.makedirs(save_dir, exist_ok=True)
    # np.save(save_dir+f"/{prompt_id}_anchor{anchor}_rnd{rnd}_iter{iters}_{init}_Adam.npy", rnd_text_emb_npy)








def main_adv_text_per_img_10_earlystopping(args):
    
    
    
    
    def encode_prompt_(caption, tokenizer, text_encoder):
        captions = [caption]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(text_encoder.device)

        encoder_hidden_states = text_encoder(input_ids)[0]
        
        return encoder_hidden_states
    
    def load_pipeline(ckpt_path, device='cuda:0'):
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe
    
    
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "runwayml/stable-diffusion-v1-5"

    ##############################################################
    # ckpt_path = "/mnt/nas5/joonsung/2025/ckpts/sd-pokemon-checkpoint/sd-pokemon-checkpoint"
    
    # ckpt_path = 'runwayml/stable-diffusion-v1-5'
    

    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.ckpt_path, subfolder="tokenizer", revision=None
    # )
    # # tokenizer = tokenizer.to(device)
    # # tokenizer = tokenizer.cuda()

    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.ckpt_path, subfolder="text_encoder", revision=None
    # )
    # text_encoder = text_encoder.to(device)

    # vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    # vae = vae.to(device)

    # unet = UNet2DConditionModel.from_pretrained(
    #     args.ckpt_path, subfolder="unet", revision=None
    # )
    # unet = unet.to(device)
    
    # text_encoder.requires_grad_(False)
    
    # for p in text_encoder.parameters():
    #     p.requires_grad = False
    
    pipe = load_pipeline(args.ckpt_path, device)
    ##############################################################
    set_random_seed(args.gen_seed)


    

    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    
    image = Image.open(args.img_path).convert("RGB")
    images = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 512, 512)
    
    prompt_id = args.img_path.split('/')[-1].split('.')[0]
    
    


        

    mse_loss = nn.MSELoss()
    
    garbage_prompt = ""
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[1].unsqueeze(0).detach()
    
    
    
    ###################################
    
    
    anchor = args.anchor
    until = args.until
    


    ## blip2
    if args.init == "ori":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/member_captions.jsonl"
        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/non_member_captions.jsonl"
        else:
             ValueError("args.mem was not satisfied.")
             
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["filename"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )
    
        init = args.init
        
    
    ## ori - mem
    elif args.init == "clip_interrogator":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_members_caption_output.jsonl"

        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_non_members_caption_output.jsonl"
            
            
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["prompt_id"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )

        init = args.init

        
        
    else:
        ValueError("args.init was not satisfied.")

    
    # noise_scale = 0.1
    optim_iters = args.OptimIter
    iters = args.iters
    eps = args.eps
    step_size = eps/2.
    GUIDANCE_SCALE = 7.5
    # epsilon = 1e-5
    # step_size > 0.01: loss increase
    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
    
    # for p in pipe.text_encoder.parameters():
    #     p.requires_grad = False
    

    
    
    
    #### 1. Adv Example ####
    
    
    set_random_seed(args.gen_seed)
    
    # pipe.unet.eval()
    # pipe.vae.eval()
    
    images = images*2. - 1.
    
    
    gen = torch.Generator(device=device).manual_seed(args.gen_seed)
    init_noise = (torch.rand(*images.shape, generator=gen, device=device, dtype=torch.float32)*2*eps - eps)
    adv_img = images.clone().detach() + init_noise
    # adv_img = images.clone().detach() + (torch.rand(*images.shape)*2*eps-eps).to(device=device, dtype=torch.float32)

    
    
    
    
    ## 1. Uncond. DDIM inversion ##
    # set_random_seed(args.gen_seed)
    # with torch.no_grad():
    #     inverted_latents = invert(pipe, anchor, latent, garbage_prompt, device=device, guidance_scale=0, num_inference_steps=50)
        
    ## 2. add_noise ##

    ## _________________________________________________________ ##
    # num_inference_steps = 100 ## SecMI setting

    # pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    ## _________________________________________________________ ##

    
    timesteps = list(range(0, 1000, 10))
    
    # trg_noise = torch.randn(inverted_latent.shape).to(device=device, dtype=torch.float32)
    

    
    # for p in pipe.unet.parameters():
    #     print(p.requires_grad) ## all True
        
    
    pbar_adv = tqdm(range(optim_iters))

    ## **** ##
    npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_1.npy"
    rnd_noise_np = np.load(npy_path)
    rnd_noise_1 = torch.from_numpy(rnd_noise_np).to(device=device, dtype=adv_img.dtype)
    
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    starter.record()   # 🔥 GPU timer start
    
    for j in range(anchor, until, -1):
        set_random_seed(args.gen_seed)
        # adv_img.requires_grad_(True)
        
        
        # optimizer = Adam([adv_img], lr=0.001) # 0.001
        # # optimizer = SGD([rnd_text_emb], lr=0.1) ## xxx 
        
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                             lr_lambda=lambda epoch: 0.95 ** epoch,
        #                             last_epoch=-1,
        #                             verbose=False)


        
        ## -------------------- old -------------------- ##
        # t=j
        # timestep = timesteps[t]
        # timestep = torch.tensor([timestep], device=device)
        
        ## -------------------- old -------------------- ##
        
        ## anchor == 20
        
        for it in pbar_adv:

            adv_img = adv_img.detach().clone().requires_grad_(True)
            
            
            ## -------------------- NEW -------------------- ##
            rnd = args.adv_rnd
            t = torch.randint(j-(rnd//2), j + (rnd//2+1), (1,)).item()  # 0 ~ t 사이의 랜덤 정수
            
            ## anchor = 130 -> {110, 120, 130, 140, 150}

            # timesteps에서 해당 값 추출
            timestep = timesteps[t]
            timestep = torch.tensor([timestep], device=device)

            ## -------------------- NEW -------------------- ##  
            
            ## 2. ##
            pipe.unet.zero_grad()
            pipe.vae.zero_grad()
            
            actual_step_size = step_size - (step_size - step_size / 100) / optim_iters * it
            # adv_latent_x0 = encode_image_grad(pipe, adv_img, dtype=torch.float32)
            
            adv_latent_x0 = pipe.vae.encode(adv_img.to(dtype=torch.float32)).latent_dist
            adv_latent_x0 = 0.18215 * adv_latent_x0.mean

            
            ## **** ##
            # rnd_noise = torch.randn(adv_latent_x0.shape).to(device=device, dtype=adv_latent_x0.dtype)
            
            adv_latent_xt = pipe.scheduler.add_noise(adv_latent_x0.to(device), rnd_noise_1.to(device), timestep)


            _, _, noise_pred_uncond, noise_pred_text = pipe.mtcnp_adv(perturb_embeds=None, perturb_latent=adv_latent_xt, prompt=pred_prompt, anchor=t, guidance_scale=7.5) ## default: 7.5



             
            ## 1. Memorization ##
            if args.type == "Memorized":
                cost = mse_loss(noise_pred_uncond, noise_pred_text)
                grad, = torch.autograd.grad(cost, [adv_img])
                adv_img = adv_img + grad.sign() * actual_step_size
            
            ## 2. Uncond ~ rnd noise ##
            elif args.type == "Uncond":
                cost = mse_loss(noise_pred_uncond, rnd_noise_1)
                grad, = torch.autograd.grad(cost, [adv_img])
                adv_img = adv_img - grad.sign() * actual_step_size
            
            
            adv_img = torch.minimum(torch.maximum(adv_img, adv_img - eps), adv_img + eps)
            adv_img.data = torch.clamp(adv_img, min=-1, max=1)
            adv_img.grad = None
            #### torch.cuda.empty_cache()

            ## 2. ## ==> ldm에서는 터짐
            # cost = (mse_loss(rnd_noise, noise_pred) / (anchor-until)).to(pipe.device)
            # cost.backward()
            # grad = adv_img.grad.detach().sign()
            # adv_img = adv_img - actual_step_size*grad
            # eta = torch.clamp(adv_img.data - images.data, min=-eps, max=eps)
            # adv_img = adv_img.detach()
            # adv_img = torch.clamp(adv_img + eta, min=-1, max=1)
            
            # if it+1 == 10:
            #     print("## ****** ##")
            #     print(f"loss at {it+1}: {cost.item():.6f}")
            #     print("## ****** ##")
            
            peak = torch.cuda.max_memory_allocated() / 1024**2

            if pbar_adv is not None:
                pbar_adv.set_description(
                    f"Image: {prompt_id} | timestep {timestep.item()} | Iter {it} | eps {eps} | peak {peak}MB --> Step size: {actual_step_size:.4f} / Current loss: {cost.item():.6f}"
                )


            
            
            ## 2. Adam
            # latent = encode_image(pipe, adv_img, dtype=torch.float32)
            # rnd_noise = torch.randn(latent.shape).to(device=device, dtype=latent.dtype)
            # latent_cur = pipe.scheduler.add_noise(latent.to(device), rnd_noise.to(device), timestep.to(device))

            # noise_pred_cond = get_noise_pred_single(pipe, latent_cur, timestep, rnd_text_emb)

    
    torch.cuda.synchronize()
    ender.record()  # 🔥 GPU timer end
    torch.cuda.synchronize()

    # Total GPU time (ms → seconds)
    gpu_time_ms = starter.elapsed_time(ender)
    gpu_time_sec = gpu_time_ms / 1000.0

    print(f"\n🔥 Total GPU time: {gpu_time_sec:.4f} seconds")
    print(f"🔥 Average per iteration: {gpu_time_sec/iters:.6f} sec/iter\n")
            

        
            

        
    
    # torch.cuda.empty_cache()
    # del grad, adv_latent_x0, adv_latent_xt, noise_pred, cost 
    
    

    torch.cuda.empty_cache()
    gc.collect()
    
    adv_img = adv_img.detach()
    

    # adv_img_cpu = adv_img.cpu().squeeze(0)  # shape: (3, H, W)
    # adv_img_cpu = (adv_img_cpu + 1) / 2  # Map to [0, 1]
    # adv_img_pil = to_pil_image(adv_img_cpu)

    ### Save as PNG
    # folder_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}"
    # if args.mem == "member":
    #     img_dir = f"{folder_dir}/members/adv_img"
    # elif args.mem == "non_member":
    #     img_dir = f"{folder_dir}/non_members/adv_img"
    # else:
    #     ValueError("args.mem was not satisfied.")

    # os.makedirs(img_dir, exist_ok=True)
    # save_path = os.path.join(img_dir, f"{prompt_id}_anchor{anchor}_rnd{rnd}_OptimIter{optim_iters}_eps{eps}_step{step_size}.png")
    # adv_img_pil.save(save_path)
    
    
    
    
    
    #### 2. Text embedding ####
    ## New
    set_random_seed(args.gen_seed)
    
    rnd_text_emb = init_text_embeddings[1].unsqueeze(0).detach() # (1, 77, 768)
    
    with torch.no_grad():
        latent = pipe.vae.encode(adv_img)
        latent_z0 = 0.18215 * latent.latent_dist.sample()
        
        cln_latent = pipe.vae.encode(images)
        cln_latent_z0 = 0.18215 * cln_latent.latent_dist.sample()


    # rnd_text_emb.requires_grad_(True)
    rnd_text_emb = rnd_text_emb.detach().clone().requires_grad_(True)
    ### From Null-text inversion ###
    # optimizer = Adam([rnd_text_emb], lr=1e-2 * (1. - t / 100.))
    optimizer = Adam([rnd_text_emb], lr=args.lr)
    
    

    def get_noise_pred_single(pipe, latents, t, context):
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=context).sample
        # noise_pred = pipe.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    
        
    # npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_2.npy"
    # rnd_noise_np = np.load(npy_path)
    # rnd_noise_2 = torch.from_numpy(rnd_noise_np).to(device)
    
    
    iters_list = list(range(0, iters + 1, 100))
    
    
    # if args.mem == "member":
    #     pre_dir = f"{folder_dir}/members"
    #     for iteration in iters_list:
    #         save_it_dir = pre_dir + f"/perturb_emb_iter{iteration}_lr{args.lr}"
    #         os.makedirs(save_it_dir, exist_ok=True)
        
    # elif args.mem == "non_member":
    #     pre_dir = f"{folder_dir}/non_members"
    #     for iteration in iters_list:
    #         save_it_dir = pre_dir + f"/perturb_emb_iter{iteration}_lr{args.lr}"
    #         os.makedirs(save_it_dir, exist_ok=True)
    # else:
    #     ValueError("args.mem was not satisfied.")
    
    pbar = tqdm(range(iters))
    for it in pbar:
        
        rnd = args.emb_rnd
        t = torch.randint(j-(rnd//2), j+(rnd//2+1), (1,)).item()  # 0 ~ t 사이의 랜덤 정수
        
        ## anchor = 150 -> {110, 120, 130, 140, 150}

        # timesteps에서 해당 값 추출
        timestep = timesteps[t]
        timestep = torch.tensor([timestep], device=device)
        
        # rnd_noise_1 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
        # rnd_noise_2 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
            
        # latent_zt_1 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        latent_zt_2 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        
        
        
        # noise_pred_uncond = get_noise_pred_single(pipe, latent_zt_2, timestep, uncond_emb.detach())
        noise_pred_cond = get_noise_pred_single(pipe, latent_zt_2, timestep, rnd_text_emb)

        

        # loss_uncond = F.mse_loss(noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
        loss_cond = F.mse_loss(noise_pred_cond.float(), rnd_noise_1.float().to(device), reduction="mean")
        
    
        # loss = (loss_cond - loss_uncond).to(pipe.device)
        loss = loss_cond.to(pipe.device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ## Eval ##
        with torch.no_grad():
            cln_latent_zt = pipe.scheduler.add_noise(cln_latent_z0.to(device), rnd_noise_1.to(device), timestep).detach()
            cln_noise_pred_uncond = get_noise_pred_single(pipe, cln_latent_zt, timestep, uncond_emb.detach())
            cln_noise_pred_cond = get_noise_pred_single(pipe, cln_latent_zt, timestep, rnd_text_emb)

            
            cln_loss_uncond = F.mse_loss(cln_noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
            cln_loss_cond = F.mse_loss(cln_noise_pred_cond.float(), rnd_noise_1.float().to(device), reduction="mean").to(device)
            
            eval_loss = (cln_loss_cond - cln_loss_uncond).to(pipe.device)
        
        
        
        
        
        if pbar is not None:
            pbar.set_description(
                # f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f} (cond: {loss_cond.item():.8f}, uncond: {loss_uncond.item():.8f})"
                f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f} ||| clean: {eval_loss.item():.8f}"
            )
        
        if (it+1) in iters_list:
            print(f"loss at iter={it+1}: {loss.item():.5f}")
            

def main_adv_text_per_img_10_Adam(args):
    
    
    
    
    def encode_prompt_(caption, tokenizer, text_encoder):
        captions = [caption]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(text_encoder.device)

        encoder_hidden_states = text_encoder(input_ids)[0]
        
        return encoder_hidden_states
    
    def load_pipeline(ckpt_path, device='cuda:0'):
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe
    
    
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "runwayml/stable-diffusion-v1-5"

    ##############################################################
    # ckpt_path = "/mnt/nas5/joonsung/2025/ckpts/sd-pokemon-checkpoint/sd-pokemon-checkpoint"
    
    # ckpt_path = 'runwayml/stable-diffusion-v1-5'
    

    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.ckpt_path, subfolder="tokenizer", revision=None
    # )
    # # tokenizer = tokenizer.to(device)
    # # tokenizer = tokenizer.cuda()

    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.ckpt_path, subfolder="text_encoder", revision=None
    # )
    # text_encoder = text_encoder.to(device)

    # vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    # vae = vae.to(device)

    # unet = UNet2DConditionModel.from_pretrained(
    #     args.ckpt_path, subfolder="unet", revision=None
    # )
    # unet = unet.to(device)
    
    # text_encoder.requires_grad_(False)
    
    # for p in text_encoder.parameters():
    #     p.requires_grad = False
    
    pipe = load_pipeline(args.ckpt_path, device)
    ##############################################################
    set_random_seed(args.gen_seed)


    

    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    
    image = Image.open(args.img_path).convert("RGB")
    images = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 512, 512)
    
    prompt_id = args.img_path.split('/')[-1].split('.')[0]
    
    


        

    mse_loss = nn.MSELoss()
    
    garbage_prompt = ""
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[1].unsqueeze(0).detach()
    
    
    
    ###################################
    
    
    anchor = args.anchor
    until = args.until
    


    ## blip2
    if args.init == "ori":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/member_captions.jsonl"
        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/non_member_captions.jsonl"
        else:
             ValueError("args.mem was not satisfied.")
             
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["filename"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )
    
        init = args.init
        
    
    ## ori - mem
    elif args.init == "clip_interrogator":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_members_caption_output.jsonl"

        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_non_members_caption_output.jsonl"
            
            
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["prompt_id"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )

        init = args.init

        
        
    else:
        ValueError("args.init was not satisfied.")

    
    # noise_scale = 0.1
    optim_iters = args.OptimIter
    iters = args.iters
    eps = args.eps
    step_size = eps/2.
    GUIDANCE_SCALE = 7.5
    # epsilon = 1e-5
    # step_size > 0.01: loss increase
    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
    
    # for p in pipe.text_encoder.parameters():
    #     p.requires_grad = False
    

    
    
    
    #### 1. Adv Example ####
    
    
    set_random_seed(args.gen_seed)
    
    # pipe.unet.eval()
    # pipe.vae.eval()
    
    images = images*2. - 1.
    
    
    gen = torch.Generator(device=device).manual_seed(args.gen_seed)
    init_noise = (torch.rand(*images.shape, generator=gen, device=device, dtype=torch.float32)*2*eps - eps)
    adv_img = images.clone().detach() + init_noise
    # adv_img = images.clone().detach() + (torch.rand(*images.shape)*2*eps-eps).to(device=device, dtype=torch.float32)

    
    
    
    
    ## 1. Uncond. DDIM inversion ##
    # set_random_seed(args.gen_seed)
    # with torch.no_grad():
    #     inverted_latents = invert(pipe, anchor, latent, garbage_prompt, device=device, guidance_scale=0, num_inference_steps=50)
        
    ## 2. add_noise ##

    ## _________________________________________________________ ##
    # num_inference_steps = 100 ## SecMI setting

    # pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    ## _________________________________________________________ ##

    
    timesteps = list(range(0, 1000, 10))
    
    # trg_noise = torch.randn(inverted_latent.shape).to(device=device, dtype=torch.float32)
    
    def get_noise_pred_single(pipe, latents, t, context):
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=context).sample
        # noise_pred = pipe.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    
    # for p in pipe.unet.parameters():
    #     print(p.requires_grad) ## all True
        
    
    pbar_adv = tqdm(range(optim_iters))

    ## **** ##
    npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_1.npy"
    rnd_noise_np = np.load(npy_path)
    rnd_noise_1 = torch.from_numpy(rnd_noise_np).to(device=device, dtype=adv_img.dtype)
    
    
    
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    starter.record()   # 🔥 GPU timer start
    
    for j in range(anchor, until, -1):
        set_random_seed(args.gen_seed)
        
        adv_img.requires_grad_(True)
        
        
        # optimizer = Adam([adv_img], lr=0.1) # 0.001
        optimizer = SGD([adv_img], lr=0.1) ## xxx 
        
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                             lr_lambda=lambda epoch: 0.95 ** epoch,
        #                             last_epoch=-1,
        #                             verbose=False)


        
        ## -------------------- old -------------------- ##
        # t=j
        # timestep = timesteps[t]
        # timestep = torch.tensor([timestep], device=device)
        
        ## -------------------- old -------------------- ##
        
        ## anchor == 20
        
        for it in pbar_adv:

            # adv_img = adv_img.detach().clone().requires_grad_(True)
            
            
            # ## -------------------- NEW -------------------- ##
            rnd = args.adv_rnd
            t = torch.randint(j-(rnd//2), j + (rnd//2+1), (1,)).item()  # 0 ~ t 사이의 랜덤 정수
            
            ## anchor = 130 -> {110, 120, 130, 140, 150}

            # timesteps에서 해당 값 추출
            timestep = timesteps[t]
            timestep = torch.tensor([timestep], device=device)

            # ## -------------------- NEW -------------------- ##  
            
            # ## 2. ##
            pipe.unet.zero_grad()
            pipe.vae.zero_grad()
            
            # actual_step_size = step_size - (step_size - step_size / 100) / optim_iters * it
            # # adv_latent_x0 = encode_image_grad(pipe, adv_img, dtype=torch.float32)
            
            # adv_latent_x0 = pipe.vae.encode(adv_img.to(dtype=torch.float32)).latent_dist
            # adv_latent_x0 = 0.18215 * adv_latent_x0.mean

            
            # ## **** ##
            # # rnd_noise = torch.randn(adv_latent_x0.shape).to(device=device, dtype=adv_latent_x0.dtype)
            
            # adv_latent_xt = pipe.scheduler.add_noise(adv_latent_x0.to(device), rnd_noise_1.to(device), timestep)


            # _, _, noise_pred_uncond, noise_pred_text = pipe.mtcnp_adv(perturb_embeds=None, perturb_latent=adv_latent_xt, prompt=pred_prompt, anchor=t, guidance_scale=7.5) ## default: 7.5



             
            # ## 1. Memorization ##
            # if args.type == "Memorized":
            #     cost = mse_loss(noise_pred_uncond, noise_pred_text)
            #     grad, = torch.autograd.grad(cost, [adv_img])
            #     adv_img = adv_img + grad.sign() * actual_step_size
            
            # ## 2. Uncond ~ rnd noise ##
            # elif args.type == "Uncond":
            #     cost = mse_loss(noise_pred_uncond, rnd_noise_1)
            #     grad, = torch.autograd.grad(cost, [adv_img])
            #     adv_img = adv_img - grad.sign() * actual_step_size
            
            
            # adv_img = torch.minimum(torch.maximum(adv_img, adv_img - eps), adv_img + eps)
            # adv_img.data = torch.clamp(adv_img, min=-1, max=1)
            # adv_img.grad = None
            #### torch.cuda.empty_cache()

            ## 2. ## ==> ldm에서는 터짐
            # cost = (mse_loss(rnd_noise, noise_pred) / (anchor-until)).to(pipe.device)
            # cost.backward()
            # grad = adv_img.grad.detach().sign()
            # adv_img = adv_img - actual_step_size*grad
            # eta = torch.clamp(adv_img.data - images.data, min=-eps, max=eps)
            # adv_img = adv_img.detach()
            # adv_img = torch.clamp(adv_img + eta, min=-1, max=1)
            
            # if it+1 == 10:
            #     print("## ****** ##")
            #     print(f"loss at {it+1}: {cost.item():.6f}")
            #     print("## ****** ##")
            
            

            # if pbar_adv is not None:
            #     pbar_adv.set_description(
            #         f"Image: {prompt_id} | timestep {timestep.item()} | Iter {it} | eps {eps} --> Step size: {actual_step_size:.4f} / Current loss: {cost.item():.6f}"
            #     )


            
            
            ## 2. Adam
            adv_latent_x0 = pipe.vae.encode(adv_img.to(dtype=torch.float32)).latent_dist
            adv_latent_x0 = 0.18215 * adv_latent_x0.mean
            
            # rnd_noise = torch.randn(latent.shape).to(device=device, dtype=latent.dtype)
            # latent_cur = pipe.scheduler.add_noise(latent.to(device), rnd_noise.to(device), timestep.to(device))
            adv_latent_xt = pipe.scheduler.add_noise(adv_latent_x0.to(device), rnd_noise_1.to(device), timestep)

            noise_pred_uncond = get_noise_pred_single(pipe, adv_latent_xt, timestep, uncond_emb)
            cost = mse_loss(noise_pred_uncond, rnd_noise_1)
            
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            
            peak = torch.cuda.max_memory_allocated() / 1024**2

            if pbar_adv is not None:
                pbar_adv.set_description(
                    f"Image: {prompt_id} | timestep {timestep.item()} | Iter {it} | eps {eps} | VRAM {peak}MB --> Current loss: {cost.item():.6f}"
                )
            

        
    
    # torch.cuda.empty_cache()
    # del grad, adv_latent_x0, adv_latent_xt, noise_pred, cost 
    
    torch.cuda.synchronize()
    ender.record()  # 🔥 GPU timer end
    torch.cuda.synchronize()

    # Total GPU time (ms → seconds)
    gpu_time_ms = starter.elapsed_time(ender)
    gpu_time_sec = gpu_time_ms / 1000.0

    print(f"\n🔥 Total GPU time: {gpu_time_sec:.4f} seconds")
    print(f"🔥 Average per iteration: {gpu_time_sec/iters:.6f} sec/iter\n")

    torch.cuda.empty_cache()
    gc.collect()
    
    adv_img = adv_img.detach()
    

    adv_img_cpu = adv_img.cpu().squeeze(0)  # shape: (3, H, W)
    adv_img_cpu = (adv_img_cpu + 1) / 2  # Map to [0, 1]
    adv_img_pil = to_pil_image(adv_img_cpu)

    ### Save as PNG
    # folder_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}"
    # if args.mem == "member":
    #     img_dir = f"{folder_dir}/members/adv_img"
    # elif args.mem == "non_member":
    #     img_dir = f"{folder_dir}/non_members/adv_img"
    # else:
    #     ValueError("args.mem was not satisfied.")

    # os.makedirs(img_dir, exist_ok=True)
    # save_path = os.path.join(img_dir, f"{prompt_id}_anchor{anchor}_rnd{rnd}_OptimIter{optim_iters}_eps{eps}_step{step_size}.png")
    # adv_img_pil.save(save_path)
    
    
    
    
    
    #### 2. Text embedding ####
    ## New
    set_random_seed(args.gen_seed)
    
    rnd_text_emb = init_text_embeddings[1].unsqueeze(0).detach() # (1, 77, 768)
    
    with torch.no_grad():
        latent = pipe.vae.encode(adv_img)
        latent_z0 = 0.18215 * latent.latent_dist.sample()
        
        cln_latent = pipe.vae.encode(images)
        cln_latent_z0 = 0.18215 * cln_latent.latent_dist.sample()


    # rnd_text_emb.requires_grad_(True)
    rnd_text_emb = rnd_text_emb.detach().clone().requires_grad_(True)
    ### From Null-text inversion ###
    # optimizer = Adam([rnd_text_emb], lr=1e-2 * (1. - t / 100.))
    optimizer = Adam([rnd_text_emb], lr=args.lr)
    
    

    
    
        
    # npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_2.npy"
    # rnd_noise_np = np.load(npy_path)
    # rnd_noise_2 = torch.from_numpy(rnd_noise_np).to(device)
    
    
    iters_list = list(range(100, iters + 1, 100))
    
    # if args.mem == "member":
    #     pre_dir = f"{folder_dir}/members"
    #     for iteration in iters_list:
    #         save_it_dir = pre_dir + f"/perturb_emb_iter{iteration}_lr{args.lr}"
    #         os.makedirs(save_it_dir, exist_ok=True)
        
    # elif args.mem == "non_member":
    #     pre_dir = f"{folder_dir}/non_members"
    #     for iteration in iters_list:
    #         save_it_dir = pre_dir + f"/perturb_emb_iter{iteration}_lr{args.lr}"
    #         os.makedirs(save_it_dir, exist_ok=True)
    # else:
    #     ValueError("args.mem was not satisfied.")
    loss_values = []
    
    pbar = tqdm(range(iters))
    for it in pbar:
        
        rnd = args.emb_rnd
        t = torch.randint(j-(rnd//2), j+(rnd//2+1), (1,)).item()  # 0 ~ t 사이의 랜덤 정수
        
        ## anchor = 150 -> {110, 120, 130, 140, 150}

        # timesteps에서 해당 값 추출
        timestep = timesteps[t]
        timestep = torch.tensor([timestep], device=device)
        
        # rnd_noise_1 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
        # rnd_noise_2 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
            
        # latent_zt_1 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        latent_zt_2 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        
        
        
        # noise_pred_uncond = get_noise_pred_single(pipe, latent_zt_2, timestep, uncond_emb.detach())
        noise_pred_cond = get_noise_pred_single(pipe, latent_zt_2, timestep, rnd_text_emb)

        

        # loss_uncond = F.mse_loss(noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
        loss_cond = F.mse_loss(noise_pred_cond.float(), rnd_noise_1.float().to(device), reduction="mean")
        
    
        # loss = (loss_cond - loss_uncond).to(pipe.device)
        loss = loss_cond.to(pipe.device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ## Eval ##
        with torch.no_grad():
            cln_latent_zt = pipe.scheduler.add_noise(cln_latent_z0.to(device), rnd_noise_1.to(device), timestep).detach()
            cln_noise_pred_uncond = get_noise_pred_single(pipe, cln_latent_zt, timestep, uncond_emb.detach())
            cln_noise_pred_cond = get_noise_pred_single(pipe, cln_latent_zt, timestep, rnd_text_emb)

            
            cln_loss_uncond = F.mse_loss(cln_noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
            cln_loss_cond = F.mse_loss(cln_noise_pred_cond.float(), rnd_noise_1.float().to(device), reduction="mean").to(device)
            
            eval_loss = (cln_loss_cond - cln_loss_uncond).to(pipe.device)
        
        
        
        
        
        if pbar is not None:
            pbar.set_description(
                # f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f} (cond: {loss_cond.item():.8f}, uncond: {loss_uncond.item():.8f})"
                f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f} ||| clean: {eval_loss.item():.8f}"
            )
        
        
        if (it+1) in iters_list:
            loss_values.append(loss.item())
            print(f"loss at iter={it+1}: {loss.item():.5f}")
            
        # if (it+1) in iters_list:
        #     rnd_text_emb_npy = rnd_text_emb.detach().cpu().numpy()
        #     save_it_dir = pre_dir + f"/perturb_emb_iter{it+1}_lr{args.lr}"
        #     np.save(save_it_dir+f"/{prompt_id}_anchor{anchor}_rnd{rnd}_iter{iters}_{init}_Adam.npy", rnd_text_emb_npy)
    
    
        
    # cond_embeddings_list.append(rnd_text_emb[:1].detach())/
    
    
    # rnd_text_emb = rnd_text_emb.detach().cpu()
    
    # rnd_text_emb_npy = rnd_text_emb.numpy()    


    ## Save .npy
    # if args.mem == "member":
    #     save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/JS_{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/perturb_emb"
    # elif args.mem == "non_member":
    #     save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/JS_{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/perturb_emb"
    # else:
    #     ValueError("args.mem was not satisfied.")

    # os.makedirs(save_dir, exist_ok=True)
    # np.save(save_dir+f"/{prompt_id}_anchor{anchor}_rnd{rnd}_iter{iters}_{init}_Adam.npy", rnd_text_emb_npy)




def main_rnd_text_11(args):
    
    
    
    
    def encode_prompt_(caption, tokenizer, text_encoder):
        captions = [caption]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(text_encoder.device)

        encoder_hidden_states = text_encoder(input_ids)[0]
        
        return encoder_hidden_states
    
    def load_pipeline(ckpt_path, device='cuda:0'):
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe
    
    
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "runwayml/stable-diffusion-v1-5"

    ##############################################################
    # ckpt_path = "/mnt/nas5/joonsung/2025/ckpts/sd-pokemon-checkpoint/sd-pokemon-checkpoint"
    
    # ckpt_path = 'runwayml/stable-diffusion-v1-5'
    

    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.ckpt_path, subfolder="tokenizer", revision=None
    # )
    # # tokenizer = tokenizer.to(device)
    # # tokenizer = tokenizer.cuda()

    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.ckpt_path, subfolder="text_encoder", revision=None
    # )
    # text_encoder = text_encoder.to(device)

    # vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    # vae = vae.to(device)

    # unet = UNet2DConditionModel.from_pretrained(
    #     args.ckpt_path, subfolder="unet", revision=None
    # )
    # unet = unet.to(device)
    
    # text_encoder.requires_grad_(False)
    
    # for p in text_encoder.parameters():
    #     p.requires_grad = False
    
    pipe = load_pipeline(args.ckpt_path, device)
    ##############################################################
    set_random_seed(args.gen_seed)


    

    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    
    image = Image.open(args.img_path).convert("RGB")
    images = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 512, 512)
    
    prompt_id = args.img_path.split('/')[-1].split('.')[0]
    
    


        

    mse_loss = nn.MSELoss()
    
    garbage_prompt = ""
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[1].unsqueeze(0).detach()
    
    
    
    ###################################
    
    
    anchor = args.anchor
    until = args.until
    


    ## blip2
    if args.init == "ori":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/member_captions.jsonl"
        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/non_member_captions.jsonl"
        else:
             ValueError("args.mem was not satisfied.")
             
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["filename"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )
    
        init = args.init
        
    
    ## ori - mem
    elif args.init == "clip_interrogator":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_members_caption_output.jsonl"

        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_non_members_caption_output.jsonl"
            
            
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["prompt_id"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )

        init = args.init

        
        
    else:
        ValueError("args.init was not satisfied.")

    
    # noise_scale = 0.1
    optim_iters = args.OptimIter
    iters = args.iters
    eps = args.eps
    step_size = eps/2.
    GUIDANCE_SCALE = 7.5
    # epsilon = 1e-5
    # step_size > 0.01: loss increase
    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
    
    # for p in pipe.text_encoder.parameters():
    #     p.requires_grad = False
    

    
    
    
    #### 1. Adv Example ####
    images = images*2. - 1.
    
    images = images.clone().detach() + (torch.rand(*images.shape)*2*eps-eps).to(device=device, dtype=torch.float32)  
    images = torch.clamp(images, -1., 1.)
    
    set_random_seed(args.gen_seed)
    
    
    
    
    #### 2. Text embedding ####
    ## New
    set_random_seed(args.gen_seed)
    
    rnd_text_emb = init_text_embeddings[1].unsqueeze(0).detach() # (1, 77, 768)
    
    with torch.no_grad():
        latent = pipe.vae.encode(images)
        latent_z0 = 0.18215 * latent.latent_dist.sample()
        
        # cln_latent = pipe.vae.encode(images)
        # cln_latent_z0 = 0.18215 * cln_latent.latent_dist.sample()


    # rnd_text_emb.requires_grad_(True)
    rnd_text_emb = rnd_text_emb.detach().clone().requires_grad_(True)
    ### From Null-text inversion ###
    # optimizer = Adam([rnd_text_emb], lr=1e-2 * (1. - t / 100.))
    optimizer = Adam([rnd_text_emb], lr=args.lr)
    
    import math

    def lr_lambda(current_step):
        warmup_steps = 50
        total_steps = 500

        if current_step < warmup_steps:
            # warmup: 0 → 1 비율
            return float(current_step) / float(max(1, warmup_steps))
        # cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def get_noise_pred_single(pipe, latents, t, context):
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=context).sample
        # noise_pred = pipe.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    
        
    # npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_2.npy"
    # rnd_noise_np = np.load(npy_path)
    # rnd_noise_2 = torch.from_numpy(rnd_noise_np).to(device)
    
    timesteps = list(range(0, 1000, 10))
    
    npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_1.npy"
    rnd_noise_np = np.load(npy_path)
    rnd_noise_1 = torch.from_numpy(rnd_noise_np).to(device)
    
    pbar = tqdm(range(iters))
    for it in pbar:
        
        rnd = args.emb_rnd
        t = torch.randint(anchor-(rnd//2), anchor+(rnd//2+1), (1,)).item()  # 0 ~ t 사이의 랜덤 정수
        
        ## anchor = 150 -> {110, 120, 130, 140, 150}

        # timesteps에서 해당 값 추출
        timestep = timesteps[t]
        timestep = torch.tensor([timestep], device=device)
        
        # rnd_noise_1 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
        # rnd_noise_2 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
            
        # latent_zt_1 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        latent_zt_2 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        
        
        
        noise_pred_uncond = get_noise_pred_single(pipe, latent_zt_2, timestep, uncond_emb.detach())
        noise_pred_cond = get_noise_pred_single(pipe, latent_zt_2, timestep, rnd_text_emb)

        

        loss_uncond = F.mse_loss(noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
        loss_cond = F.mse_loss(noise_pred_cond.float(), rnd_noise_1.float().to(device), reduction="mean")
        
    
        loss = (loss_cond - loss_uncond).to(pipe.device)
        # loss = loss_cond.to(pipe.device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        ## Eval ##
        # with torch.no_grad():
        #     cln_latent_zt = pipe.scheduler.add_noise(cln_latent_z0.to(device), rnd_noise_1.to(device), timestep).detach()
        #     cln_noise_pred_uncond = get_noise_pred_single(pipe, cln_latent_zt, timestep, uncond_emb.detach())
        #     cln_noise_pred_cond = get_noise_pred_single(pipe, cln_latent_zt, timestep, rnd_text_emb)

            
        #     cln_loss_uncond = F.mse_loss(cln_noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
        #     cln_loss_cond = F.mse_loss(cln_noise_pred_cond.float(), rnd_noise_1.float().to(device), reduction="mean").to(device)
            
        #     eval_loss = (cln_loss_cond - cln_loss_uncond).to(pipe.device)
        
        
        
        
        if pbar is not None:
            pbar.set_description(
                # f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f} (cond: {loss_cond.item():.8f}, uncond: {loss_uncond.item():.8f})"
                f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f}"
            )
        
    # cond_embeddings_list.append(rnd_text_emb[:1].detach())/
    
    
    rnd_text_emb = rnd_text_emb.detach().cpu()
    
    rnd_text_emb_npy = rnd_text_emb.numpy()    


    ## Save .npy
    
    if args.mem == "member":
        save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/{args.type}_anchor{anchor}_init_{init}_iters{iters}_eps{eps}/members/perturb_emb"
    elif args.mem == "non_member":
        save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/{args.type}_anchor{anchor}_init_{init}_iters{iters}_eps{eps}/non_members/perturb_emb"
    else:
        ValueError("args.mem was not satisfied.")

    os.makedirs(save_dir, exist_ok=True)
    np.save(save_dir+f"/{prompt_id}_anchor{anchor}_rnd{rnd}_iter{iters}_{init}_Adam.npy", rnd_text_emb_npy)


    




def main_adv_text_per_img_12(args):
    
    
    
    
    def encode_prompt_(caption, tokenizer, text_encoder):
        captions = [caption]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(text_encoder.device)

        encoder_hidden_states = text_encoder(input_ids)[0]
        
        return encoder_hidden_states
    
    def load_pipeline(ckpt_path, device='cuda:0'):
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe
    
    
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "runwayml/stable-diffusion-v1-5"

    ##############################################################
    # ckpt_path = "/mnt/nas5/joonsung/2025/ckpts/sd-pokemon-checkpoint/sd-pokemon-checkpoint"
    
    # ckpt_path = 'runwayml/stable-diffusion-v1-5'
    

    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.ckpt_path, subfolder="tokenizer", revision=None
    # )
    # # tokenizer = tokenizer.to(device)
    # # tokenizer = tokenizer.cuda()

    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.ckpt_path, subfolder="text_encoder", revision=None
    # )
    # text_encoder = text_encoder.to(device)

    # vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    # vae = vae.to(device)

    # unet = UNet2DConditionModel.from_pretrained(
    #     args.ckpt_path, subfolder="unet", revision=None
    # )
    # unet = unet.to(device)
    
    # text_encoder.requires_grad_(False)
    
    # for p in text_encoder.parameters():
    #     p.requires_grad = False
    
    pipe = load_pipeline(args.ckpt_path, device)
    ##############################################################
    set_random_seed(args.gen_seed)


    

    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    
    image = Image.open(args.img_path).convert("RGB")
    images = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 512, 512)
    
    prompt_id = args.img_path.split('/')[-1].split('.')[0]
    
    


        

    mse_loss = nn.MSELoss()
    
    garbage_prompt = ""
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[1].unsqueeze(0).detach()
    
    
    
    ###################################
    
    
    anchor = args.anchor
    until = args.until
    


    ## blip2
    if args.init == "ori":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/member_captions.jsonl"
        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/non_member_captions.jsonl"
        else:
             ValueError("args.mem was not satisfied.")
             
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["filename"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )
    
        init = args.init
        
    
    ## ori - mem
    elif args.init == "clip_interrogator":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_members_caption_output.jsonl"

        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_non_members_caption_output.jsonl"
            
            
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["prompt_id"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )

        init = args.init

        
        
    else:
        ValueError("args.init was not satisfied.")

    
    # noise_scale = 0.1
    optim_iters = args.OptimIter
    iters = args.iters
    eps = args.eps
    step_size = eps/2.
    GUIDANCE_SCALE = 7.5

    # epsilon = 1e-5
    # step_size > 0.01: loss increase
    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
    
    # for p in pipe.text_encoder.parameters():
    #     p.requires_grad = False
    

    
    
    
    #### 1. Adv Example ####
    images = images*2. - 1.
    
    set_random_seed(args.gen_seed)
    adv_img = images.clone().detach() + (torch.rand(*images.shape)*2*eps-eps).to(device=device, dtype=torch.float32)

    
    
    
    
    ## 1. Uncond. DDIM inversion ##
    # set_random_seed(args.gen_seed)
    # with torch.no_grad():
    #     inverted_latents = invert(pipe, anchor, latent, garbage_prompt, device=device, guidance_scale=0, num_inference_steps=50)
        
    ## 2. add_noise ##

    ## _________________________________________________________ ##
    # num_inference_steps = 100 ## SecMI setting

    # pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    ## _________________________________________________________ ##

    
    timesteps = list(range(0, 1000, 10))
    
    # trg_noise = torch.randn(inverted_latent.shape).to(device=device, dtype=torch.float32)
    

    
    # for p in pipe.unet.parameters():
    #     print(p.requires_grad) ## all True
        
    
    pbar_adv = tqdm(range(optim_iters))

    ## **** ##
    npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_1.npy"
    rnd_noise_np = np.load(npy_path)
    rnd_noise_1 = torch.from_numpy(rnd_noise_np).to(device)
    
    for j in range(anchor, until, -1):
        set_random_seed(args.gen_seed)
        # adv_img.requires_grad_(True)
        
        
        # optimizer = Adam([adv_img], lr=0.001) # 0.001
        # # optimizer = SGD([rnd_text_emb], lr=0.1) ## xxx 
        
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                             lr_lambda=lambda epoch: 0.95 ** epoch,
        #                             last_epoch=-1,
        #                             verbose=False)


        
        ## -------------------- old -------------------- ##
        # t=j
        # timestep = timesteps[t]
        # timestep = torch.tensor([timestep], device=device)
        
        ## -------------------- old -------------------- ##
        
        ## anchor == 20
        
        for it in pbar_adv:

            adv_img = adv_img.detach().clone().requires_grad_(True)
            
            
            ## -------------------- NEW -------------------- ##
            rnd = args.adv_rnd
            t = torch.randint(j-(rnd//2), j + (rnd//2+1), (1,)).item()  # 0 ~ t 사이의 랜덤 정수
            
            ## anchor = 130 -> {110, 120, 130, 140, 150}

            # timesteps에서 해당 값 추출
            timestep = timesteps[t]
            timestep = torch.tensor([timestep], device=device)

            ## -------------------- NEW -------------------- ##  
            
            ## 2. ##
            pipe.unet.zero_grad()
            pipe.vae.zero_grad()
            
            actual_step_size = step_size - (step_size - step_size / 100) / optim_iters * it
            # adv_latent_x0 = encode_image_grad(pipe, adv_img, dtype=torch.float32)
            
            adv_latent_x0 = pipe.vae.encode(adv_img.to(dtype=torch.float32))
            adv_latent_x0 = 0.18215 * adv_latent_x0.latent_dist.sample()

            
            ## **** ##
            # rnd_noise = torch.randn(adv_latent_x0.shape).to(device=device, dtype=adv_latent_x0.dtype)
            
            adv_latent_xt = pipe.scheduler.add_noise(adv_latent_x0.to(device), rnd_noise_1.to(device), timestep)


            _, _, noise_pred_uncond, noise_pred_text = pipe.mtcnp_adv(perturb_embeds=None, perturb_latent=adv_latent_xt, prompt=pred_prompt, anchor=t, guidance_scale=7.5) ## default: 7.5


            

            pipe.unet.zero_grad()
             
            ## 1. Memorization ##
            if args.type == "Memorized":
                cost = mse_loss(noise_pred_uncond, noise_pred_text)
                grad, = torch.autograd.grad(cost, [adv_img])
                adv_img = adv_img + grad.sign() * actual_step_size
            
            ## 2. Uncond ~ rnd noise ##
            elif args.type == "Uncond":
                cost = mse_loss(noise_pred_uncond, rnd_noise_1)
                grad, = torch.autograd.grad(cost, [adv_img])
                adv_img = adv_img - grad.sign() * actual_step_size
            
            
            adv_img = torch.minimum(torch.maximum(adv_img, adv_img - eps), adv_img + eps)
            adv_img.data = torch.clamp(adv_img, min=-1, max=1)
            adv_img.grad = None
            #### torch.cuda.empty_cache()

            ## 2. ## ==> ldm에서는 터짐
            # cost = (mse_loss(rnd_noise, noise_pred) / (anchor-until)).to(pipe.device)
            # cost.backward()
            # grad = adv_img.grad.detach().sign()
            # adv_img = adv_img - actual_step_size*grad
            # eta = torch.clamp(adv_img.data - images.data, min=-eps, max=eps)
            # adv_img = adv_img.detach()
            # adv_img = torch.clamp(adv_img + eta, min=-1, max=1)
            
            
            

            if pbar_adv is not None:
                pbar_adv.set_description(
                    f"Image: {prompt_id} | timestep {timestep.item()} | Iter {it} | eps {eps} --> Step size: {actual_step_size:.4f} / Current loss: {cost.item():.6f}"
                )


            ## 2. Adam
            # latent = encode_image(pipe, adv_img, dtype=torch.float32)
            # rnd_noise = torch.randn(latent.shape).to(device=device, dtype=latent.dtype)
            # latent_cur = pipe.scheduler.add_noise(latent.to(device), rnd_noise.to(device), timestep.to(device))

            # noise_pred_cond = get_noise_pred_single(pipe, latent_cur, timestep, rnd_text_emb)


        
            

        
    
    # torch.cuda.empty_cache()
    # del grad, adv_latent_x0, adv_latent_xt, noise_pred, cost 
    
    

    torch.cuda.empty_cache()
    gc.collect()
    
    adv_img = adv_img.detach()
    

    adv_img_cpu = adv_img.cpu().squeeze(0)  # shape: (3, H, W)
    adv_img_cpu = (adv_img_cpu + 1) / 2  # Map to [0, 1]
    adv_img_pil = to_pil_image(adv_img_cpu)

    ### Save as PNG        
    if args.mem == "member":
        img_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/getBEST_{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}_lr{args.lr}/members/adv_img"
    elif args.mem == "non_member":
        img_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/getBEST_{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}_lr{args.lr}/non_members/adv_img"
    else:
        ValueError("args.mem was not satisfied.")

    os.makedirs(img_dir, exist_ok=True)
    save_path = os.path.join(img_dir, f"{prompt_id}_anchor{anchor}_rnd{rnd}_OptimIter{optim_iters}_eps{eps}_step{step_size}.png")
    adv_img_pil.save(save_path)
    
    
    
    
    
    #### 2. Text embedding ####
    ## New
    set_random_seed(args.gen_seed)
    
    rnd_text_emb = init_text_embeddings[1].unsqueeze(0).detach() # (1, 77, 768)
    
    with torch.no_grad():
        latent = pipe.vae.encode(adv_img)
        latent_z0 = 0.18215 * latent.latent_dist.sample()
        
        cln_latent = pipe.vae.encode(images)
        cln_latent_z0 = 0.18215 * cln_latent.latent_dist.sample()


    # rnd_text_emb.requires_grad_(True)
    rnd_text_emb = rnd_text_emb.detach().clone().requires_grad_(True)
    ### From Null-text inversion ###
    # optimizer = Adam([rnd_text_emb], lr=1e-2 * (1. - t / 100.))
    optimizer = Adam([rnd_text_emb], lr=args.lr)
    
    

    def get_noise_pred_single(pipe, latents, t, context):
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=context).sample
        # noise_pred = pipe.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    
        
    # npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_2.npy"
    # rnd_noise_np = np.load(npy_path)
    # rnd_noise_2 = torch.from_numpy(rnd_noise_np).to(device)
    
    
    
    pbar = tqdm(range(iters))
    for it in pbar:
        
        rnd = args.emb_rnd
        t = torch.randint(j-(rnd//2), j+(rnd//2+1), (1,)).item()  # 0 ~ t 사이의 랜덤 정수
        
        ## anchor = 150 -> {110, 120, 130, 140, 150}

        # timesteps에서 해당 값 추출
        timestep = timesteps[t]
        timestep = torch.tensor([timestep], device=device)
        
        # rnd_noise_1 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
        # rnd_noise_2 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
            
        latent_zt_1 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        latent_zt_2 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        
        
        
        noise_pred_uncond = get_noise_pred_single(pipe, latent_zt_1, timestep, uncond_emb.detach())
        noise_pred_cond = get_noise_pred_single(pipe, latent_zt_2, timestep, rnd_text_emb)

        

        loss_uncond = F.mse_loss(noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
        loss_cond = F.mse_loss(noise_pred_cond.float(), rnd_noise_1.float().to(device), reduction="mean")
        
    
        loss = (loss_cond - loss_uncond).to(pipe.device)
        # loss = loss_cond.to(pipe.device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ## Eval ##
        with torch.no_grad():
            cln_latent_zt = pipe.scheduler.add_noise(cln_latent_z0.to(device), rnd_noise_1.to(device), timestep).detach()
            cln_noise_pred_uncond = get_noise_pred_single(pipe, cln_latent_zt, timestep, uncond_emb.detach())
            cln_noise_pred_cond = get_noise_pred_single(pipe, cln_latent_zt, timestep, rnd_text_emb)

            
            cln_loss_uncond = F.mse_loss(cln_noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
            cln_loss_cond = F.mse_loss(cln_noise_pred_cond.float(), rnd_noise_1.float().to(device), reduction="mean").to(device)
            
            eval_loss = (cln_loss_cond - cln_loss_uncond).to(pipe.device)
        
        
        
        
        if pbar is not None:
            pbar.set_description(
                # f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f} (cond: {loss_cond.item():.8f}, uncond: {loss_uncond.item():.8f})"
                f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f} ||| cln_cond:{cln_loss_cond.item():.5f}, cln_uncond:{cln_loss_uncond.item():.5f}, clean: {eval_loss.item():.5f}"
            )
        
    # cond_embeddings_list.append(rnd_text_emb[:1].detach())/
    
    
    rnd_text_emb = rnd_text_emb.detach().cpu()
    
    rnd_text_emb_npy = rnd_text_emb.numpy()    


    ## Save .npy
    
    if args.mem == "member":
        save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/getBEST_{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}_lr{args.lr}/members/perturb_emb"
    elif args.mem == "non_member":
        save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/getBEST_{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}_lr{args.lr}/non_members/perturb_emb"
    else:
        ValueError("args.mem was not satisfied.")

    os.makedirs(save_dir, exist_ok=True)
    np.save(save_dir+f"/{prompt_id}_anchor{anchor}_rnd{rnd}_iter{iters}_{init}_Adam.npy", rnd_text_emb_npy)





def main_adv_text_per_img_13(args):
    
    
    
    
    def encode_prompt_(caption, tokenizer, text_encoder):
        captions = [caption]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(text_encoder.device)

        encoder_hidden_states = text_encoder(input_ids)[0]
        
        return encoder_hidden_states
    
    def load_pipeline(ckpt_path, device='cuda:0'):
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe
    
    
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "runwayml/stable-diffusion-v1-5"

    ##############################################################
    # ckpt_path = "/mnt/nas5/joonsung/2025/ckpts/sd-pokemon-checkpoint/sd-pokemon-checkpoint"
    
    # ckpt_path = 'runwayml/stable-diffusion-v1-5'
    

    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.ckpt_path, subfolder="tokenizer", revision=None
    # )
    # # tokenizer = tokenizer.to(device)
    # # tokenizer = tokenizer.cuda()

    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.ckpt_path, subfolder="text_encoder", revision=None
    # )
    # text_encoder = text_encoder.to(device)

    # vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    # vae = vae.to(device)

    # unet = UNet2DConditionModel.from_pretrained(
    #     args.ckpt_path, subfolder="unet", revision=None
    # )
    # unet = unet.to(device)
    
    # text_encoder.requires_grad_(False)
    
    # for p in text_encoder.parameters():
    #     p.requires_grad = False
    
    pipe = load_pipeline(args.ckpt_path, device)
    ##############################################################
    set_random_seed(args.gen_seed)


    

    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    
    image = Image.open(args.img_path).convert("RGB")
    images = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 512, 512)
    
    prompt_id = args.img_path.split('/')[-1].split('.')[0]
    
    


        

    mse_loss = nn.MSELoss()
    
    garbage_prompt = ""
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[1].unsqueeze(0).detach()
    
    
    
    ###################################
    
    
    anchor = args.anchor
    until = args.until
    


    ## blip2
    if args.init == "ori":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/member_captions.jsonl"
        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/non_member_captions.jsonl"
        else:
             ValueError("args.mem was not satisfied.")
             
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["filename"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )
    
        init = args.init
        
    
    ## ori - mem
    elif args.init == "clip_interrogator":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_members_caption_output.jsonl"

        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_non_members_caption_output.jsonl"
            
            
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["prompt_id"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )

        init = args.init

        
        
    else:
        ValueError("args.init was not satisfied.")

    
    # noise_scale = 0.1
    optim_iters = args.OptimIter
    iters = args.iters
    eps = args.eps
    step_size = eps/2.
    GUIDANCE_SCALE = 7.5
    # epsilon = 1e-5
    # step_size > 0.01: loss increase
    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
    
    # for p in pipe.text_encoder.parameters():
    #     p.requires_grad = False
    

    
    
    
    #### 1. Adv Example ####
    
    
    set_random_seed(args.gen_seed)
    
    # pipe.unet.eval()
    # pipe.vae.eval()
    
    images = images*2. - 1.
    
    
    gen = torch.Generator(device=device).manual_seed(args.gen_seed)
    init_noise = (torch.rand(*images.shape, generator=gen, device=device, dtype=torch.float32)*2*eps - eps)
    adv_img = images.clone().detach() + init_noise
    # adv_img = images.clone().detach() + (torch.rand(*images.shape)*2*eps-eps).to(device=device, dtype=torch.float32)

    
    
    
    
    ## 1. Uncond. DDIM inversion ##
    # set_random_seed(args.gen_seed)
    # with torch.no_grad():
    #     inverted_latents = invert(pipe, anchor, latent, garbage_prompt, device=device, guidance_scale=0, num_inference_steps=50)
        
    ## 2. add_noise ##

    ## _________________________________________________________ ##
    # num_inference_steps = 100 ## SecMI setting

    # pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    ## _________________________________________________________ ##

    
    timesteps = list(range(0, 1000, 10))
    
    # trg_noise = torch.randn(inverted_latent.shape).to(device=device, dtype=torch.float32)
    

    
    # for p in pipe.unet.parameters():
    #     print(p.requires_grad) ## all True
        
    
    pbar_adv = tqdm(range(optim_iters))

    ## **** ##
    npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_1.npy"
    rnd_noise_np = np.load(npy_path)
    rnd_noise_1 = torch.from_numpy(rnd_noise_np).to(device=device, dtype=adv_img.dtype)
    
    for j in range(anchor, until, -1):
        set_random_seed(args.gen_seed)
        # adv_img.requires_grad_(True)
        
        
        # optimizer = Adam([adv_img], lr=0.001) # 0.001
        # # optimizer = SGD([rnd_text_emb], lr=0.1) ## xxx 
        
        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
        #                             lr_lambda=lambda epoch: 0.95 ** epoch,
        #                             last_epoch=-1,
        #                             verbose=False)


        
        ## -------------------- old -------------------- ##
        # t=j
        # timestep = timesteps[t]
        # timestep = torch.tensor([timestep], device=device)
        
        ## -------------------- old -------------------- ##
        
        ## anchor == 20
        
        for it in pbar_adv:

            adv_img = adv_img.detach().clone().requires_grad_(True)
            
            
            ## -------------------- NEW -------------------- ##
            rnd = args.adv_rnd
            t = torch.randint(j-(rnd//2), j + (rnd//2+1), (1,)).item()  # 0 ~ t 사이의 랜덤 정수
            
            ## anchor = 130 -> {110, 120, 130, 140, 150}

            # timesteps에서 해당 값 추출
            timestep = timesteps[t]
            timestep = torch.tensor([timestep], device=device)

            ## -------------------- NEW -------------------- ##  
            
            ## 2. ##
            pipe.unet.zero_grad()
            pipe.vae.zero_grad()
            
            actual_step_size = step_size - (step_size - step_size / 100) / optim_iters * it
            # adv_latent_x0 = encode_image_grad(pipe, adv_img, dtype=torch.float32)
            
            adv_latent_x0 = pipe.vae.encode(adv_img.to(dtype=torch.float32)).latent_dist
            adv_latent_x0 = 0.18215 * adv_latent_x0.mean

            
            ## **** ##
            # rnd_noise = torch.randn(adv_latent_x0.shape).to(device=device, dtype=adv_latent_x0.dtype)
            
            adv_latent_xt = pipe.scheduler.add_noise(adv_latent_x0.to(device), rnd_noise_1.to(device), timestep)


            _, _, noise_pred_uncond, noise_pred_text = pipe.mtcnp_adv(perturb_embeds=None, perturb_latent=adv_latent_xt, prompt=pred_prompt, anchor=t, guidance_scale=7.5) ## default: 7.5



             
            ## 1. Memorization ##
            if args.type == "Memorized":
                cost = mse_loss(noise_pred_uncond, noise_pred_text)
                grad, = torch.autograd.grad(cost, [adv_img])
                adv_img = adv_img + grad.sign() * actual_step_size
            
            ## 2. Uncond ~ rnd noise ##
            elif args.type == "Uncond":
                cost = mse_loss(noise_pred_uncond, rnd_noise_1)
                grad, = torch.autograd.grad(cost, [adv_img])
                adv_img = adv_img + grad.sign() * actual_step_size
            
            
            adv_img = torch.minimum(torch.maximum(adv_img, adv_img - eps), adv_img + eps)
            adv_img.data = torch.clamp(adv_img, min=-1, max=1)
            adv_img.grad = None
            #### torch.cuda.empty_cache()

            ## 2. ## ==> ldm에서는 터짐
            # cost = (mse_loss(rnd_noise, noise_pred) / (anchor-until)).to(pipe.device)
            # cost.backward()
            # grad = adv_img.grad.detach().sign()
            # adv_img = adv_img - actual_step_size*grad
            # eta = torch.clamp(adv_img.data - images.data, min=-eps, max=eps)
            # adv_img = adv_img.detach()
            # adv_img = torch.clamp(adv_img + eta, min=-1, max=1)
            
            # if it+1 == 10:
            #     print("## ****** ##")
            #     print(f"loss at {it+1}: {cost.item():.6f}")
            #     print("## ****** ##")
            
            

            if pbar_adv is not None:
                pbar_adv.set_description(
                    f"Image: {prompt_id} | timestep {timestep.item()} | Iter {it} | eps {eps} --> Step size: {actual_step_size:.4f} / Current loss: {cost.item():.6f}"
                )


            
            
            ## 2. Adam
            # latent = encode_image(pipe, adv_img, dtype=torch.float32)
            # rnd_noise = torch.randn(latent.shape).to(device=device, dtype=latent.dtype)
            # latent_cur = pipe.scheduler.add_noise(latent.to(device), rnd_noise.to(device), timestep.to(device))

            # noise_pred_cond = get_noise_pred_single(pipe, latent_cur, timestep, rnd_text_emb)


        
            

        
    
    # torch.cuda.empty_cache()
    # del grad, adv_latent_x0, adv_latent_xt, noise_pred, cost 
    
    

    torch.cuda.empty_cache()
    gc.collect()
    
    adv_img = adv_img.detach()
    

    adv_img_cpu = adv_img.cpu().squeeze(0)  # shape: (3, H, W)
    adv_img_cpu = (adv_img_cpu + 1) / 2  # Map to [0, 1]
    adv_img_pil = to_pil_image(adv_img_cpu)

    ### Save as PNG        
    # if args.mem == "member":
    #     img_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/JS_{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/adv_img"
    # elif args.mem == "non_member":
    #     img_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/JS_{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/adv_img"
    # else:
    #     ValueError("args.mem was not satisfied.")

    # os.makedirs(img_dir, exist_ok=True)
    # save_path = os.path.join(img_dir, f"{prompt_id}_anchor{anchor}_rnd{rnd}_OptimIter{optim_iters}_eps{eps}_step{step_size}.png")
    # adv_img_pil.save(save_path)
    
    
    
    
    
    #### 2. Text embedding ####
    ## New
    set_random_seed(args.gen_seed)
    
    rnd_text_emb = init_text_embeddings[1].unsqueeze(0).detach() # (1, 77, 768)
    
    with torch.no_grad():
        latent = pipe.vae.encode(adv_img)
        latent_z0 = 0.18215 * latent.latent_dist.sample()
        
        cln_latent = pipe.vae.encode(images)
        cln_latent_z0 = 0.18215 * cln_latent.latent_dist.sample()


    # rnd_text_emb.requires_grad_(True)
    rnd_text_emb = rnd_text_emb.detach().clone().requires_grad_(True)
    ### From Null-text inversion ###
    # optimizer = Adam([rnd_text_emb], lr=1e-2 * (1. - t / 100.))
    optimizer = Adam([rnd_text_emb], lr=args.lr)
    
    

    def get_noise_pred_single(pipe, latents, t, context):
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=context).sample
        # noise_pred = pipe.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    
        
    # npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_2.npy"
    # rnd_noise_np = np.load(npy_path)
    # rnd_noise_2 = torch.from_numpy(rnd_noise_np).to(device)
    
    
    
    pbar = tqdm(range(iters))
    for it in pbar:
        
        rnd = args.emb_rnd
        t = torch.randint(j-(rnd//2), j+(rnd//2+1), (1,)).item()  # 0 ~ t 사이의 랜덤 정수
        
        ## anchor = 150 -> {110, 120, 130, 140, 150}

        # timesteps에서 해당 값 추출
        timestep = timesteps[t]
        timestep = torch.tensor([timestep], device=device)
        
        # rnd_noise_1 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
        # rnd_noise_2 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
            
        # latent_zt_1 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        latent_zt_2 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        
        
        
        # noise_pred_uncond = get_noise_pred_single(pipe, latent_zt_2, timestep, uncond_emb.detach())
        noise_pred_cond = get_noise_pred_single(pipe, latent_zt_2, timestep, rnd_text_emb)

        

        # loss_uncond = F.mse_loss(noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
        loss_cond = F.mse_loss(noise_pred_cond.float(), rnd_noise_1.float().to(device), reduction="mean")
        
    
        # loss = (loss_cond - loss_uncond).to(pipe.device)
        loss = loss_cond.to(pipe.device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ## Eval ##
        with torch.no_grad():
            cln_latent_zt = pipe.scheduler.add_noise(cln_latent_z0.to(device), rnd_noise_1.to(device), timestep).detach()
            cln_noise_pred_uncond = get_noise_pred_single(pipe, cln_latent_zt, timestep, uncond_emb.detach())
            cln_noise_pred_cond = get_noise_pred_single(pipe, cln_latent_zt, timestep, rnd_text_emb)

            
            cln_loss_uncond = F.mse_loss(cln_noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
            cln_loss_cond = F.mse_loss(cln_noise_pred_cond.float(), rnd_noise_1.float().to(device), reduction="mean").to(device)
            
            eval_loss = (cln_loss_cond - cln_loss_uncond).to(pipe.device)
        
        
        
        
        if pbar is not None:
            pbar.set_description(
                # f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f} (cond: {loss_cond.item():.8f}, uncond: {loss_uncond.item():.8f})"
                f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f} ||| clean: {eval_loss.item():.8f}"
            )
        
    # cond_embeddings_list.append(rnd_text_emb[:1].detach())/
    
    
    rnd_text_emb = rnd_text_emb.detach().cpu()
    
    rnd_text_emb_npy = rnd_text_emb.numpy()    


    ## Save .npy
    
    # if args.mem == "member":
    #     save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/JS_{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/perturb_emb"
    # elif args.mem == "non_member":
    #     save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/JS_{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/perturb_emb"
    # else:
    #     ValueError("args.mem was not satisfied.")

    # os.makedirs(save_dir, exist_ok=True)
    # np.save(save_dir+f"/{prompt_id}_anchor{anchor}_rnd{rnd}_iter{iters}_{init}_Adam.npy", rnd_text_emb_npy)







def main_text(args):
    
    
    
    
    def encode_prompt_(caption, tokenizer, text_encoder):
        captions = [caption]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(text_encoder.device)

        encoder_hidden_states = text_encoder(input_ids)[0]
        
        return encoder_hidden_states
    
    def load_pipeline(ckpt_path, device='cuda:0'):
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe
    
    
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "runwayml/stable-diffusion-v1-5"

    ##############################################################
    # ckpt_path = "/mnt/nas5/joonsung/2025/ckpts/sd-pokemon-checkpoint/sd-pokemon-checkpoint"
    
    # ckpt_path = 'runwayml/stable-diffusion-v1-5'
    

    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.ckpt_path, subfolder="tokenizer", revision=None
    # )
    # # tokenizer = tokenizer.to(device)
    # # tokenizer = tokenizer.cuda()

    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.ckpt_path, subfolder="text_encoder", revision=None
    # )
    # text_encoder = text_encoder.to(device)

    # vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    # vae = vae.to(device)

    # unet = UNet2DConditionModel.from_pretrained(
    #     args.ckpt_path, subfolder="unet", revision=None
    # )
    # unet = unet.to(device)
    
    # text_encoder.requires_grad_(False)
    
    # for p in text_encoder.parameters():
    #     p.requires_grad = False
    
    pipe = load_pipeline(args.ckpt_path, device)
    ##############################################################
    set_random_seed(args.gen_seed)


    

    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    
    image = Image.open(args.img_path).convert("RGB")
    images = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 512, 512)
    
    prompt_id = args.img_path.split('/')[-1].split('.')[0]
    
    


        

    mse_loss = nn.MSELoss()
    
    garbage_prompt = ""
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[1].unsqueeze(0).detach()
    
    
    
    ###################################
    
    
    anchor = args.anchor
    until = args.until
    


    ## blip2
    if args.init == "ori":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/member_captions.jsonl"
        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/non_member_captions.jsonl"
        else:
             ValueError("args.mem was not satisfied.")
             
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["filename"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )
    
        init = args.init
        
    
    ## ori - mem
    elif args.init == "clip_interrogator":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_members_caption_output.jsonl"

        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_non_members_caption_output.jsonl"
            
            
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["prompt_id"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )

        init = args.init

        
        
    else:
        ValueError("args.init was not satisfied.")

    
    # noise_scale = 0.1
    optim_iters = args.OptimIter
    iters = args.iters
    eps = args.eps
    step_size = eps/2.
    GUIDANCE_SCALE = 7.5
    # epsilon = 1e-5
    # step_size > 0.01: loss increase
    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
    
    # for p in pipe.text_encoder.parameters():
    #     p.requires_grad = False
    

    
    
    
    #### 1. Adv Example ####
    images = images*2. - 1.
    
    set_random_seed(args.gen_seed)
    
    
    
    
    #### 2. Text embedding ####
    ## New
    set_random_seed(args.gen_seed)
    
    rnd_text_emb = init_text_embeddings[1].unsqueeze(0).detach() # (1, 77, 768)
    
    with torch.no_grad():
        latent = pipe.vae.encode(images)
        latent_z0 = 0.18215 * latent.latent_dist.sample()


    # rnd_text_emb.requires_grad_(True)
    rnd_text_emb = rnd_text_emb.detach().clone().requires_grad_(True)
    ### From Null-text inversion ###
    # optimizer = Adam([rnd_text_emb], lr=1e-2 * (1. - t / 100.))
    optimizer = Adam([rnd_text_emb], lr=1e-2)
    
    

    def get_noise_pred_single(pipe, latents, t, context):
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=context).sample
        # noise_pred = pipe.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    
        
    npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_2.npy"
    rnd_noise_np = np.load(npy_path)
    rnd_noise_2 = torch.from_numpy(rnd_noise_np).to(device)
    
    timesteps = list(range(0, 1000, 10))
    
    pbar = tqdm(range(iters))
    for it in pbar:
        
        rnd = args.emb_rnd
        t = torch.randint(anchor-(rnd//2), anchor+(rnd//2+1), (1,)).item()  # 0 ~ t 사이의 랜덤 정수
        
        ## anchor = 150 -> {110, 120, 130, 140, 150}

        # timesteps에서 해당 값 추출
        timestep = timesteps[t]
        timestep = torch.tensor([timestep], device=device)
        
        # rnd_noise_1 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
        # rnd_noise_2 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
            
        # latent_zt_1 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        latent_zt_2 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_2.to(device), timestep)
        
        # noise_pred_uncond = get_noise_pred_single(pipe, latent_zt_1, timestep, uncond_emb.detach())
        noise_pred_cond = get_noise_pred_single(pipe, latent_zt_2, timestep, rnd_text_emb)

        

        # loss_uncond = F.mse_loss(noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
        loss_cond = F.mse_loss(noise_pred_cond.float(), rnd_noise_2.float().to(device), reduction="mean")
        
    
        # loss = (loss_cond - loss_uncond).to(pipe.device)
        
        loss = loss_cond.to(pipe.device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ## Early Stopping ##
        # loss_item = loss.item()
        # if loss_item < epsilon + t * 2e-5:
        #     break
        
        if pbar is not None:
            pbar.set_description(
                # f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f} (cond: {loss_cond.item():.8f}, uncond: {loss_uncond.item():.8f})"
                f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f}"
            )
        
    # cond_embeddings_list.append(rnd_text_emb[:1].detach())/
    
    
    rnd_text_emb = rnd_text_emb.detach().cpu()
    
    rnd_text_emb_npy = rnd_text_emb.numpy()    


    ## Save .npy
    # if args.mem == "member":
    #     save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/perturb_emb"
    # elif args.mem == "non_member":
    #     save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/perturb_emb"
    # else:
    #     ValueError("args.mem was not satisfied.")

    # os.makedirs(save_dir, exist_ok=True)
    
    
    if args.mem == "member":
        save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/Test_{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/members/perturb_emb"
    elif args.mem == "non_member":
        save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/Test_{args.type}_anchor{anchor}_init_{init}_OptimIter{optim_iters}_iters{iters}_eps{eps}/non_members/perturb_emb"
    else:
        ValueError("args.mem was not satisfied.")

    os.makedirs(save_dir, exist_ok=True)
    np.save(save_dir+f"/{prompt_id}_anchor{anchor}_rnd{rnd}_iter{iters}_{init}_Adam.npy", rnd_text_emb_npy)
    



   

def main_text_emb_pool(args):
    
    
    
    
    def encode_prompt_(caption, tokenizer, text_encoder):
        captions = [caption]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(text_encoder.device)

        encoder_hidden_states = text_encoder(input_ids)[0]
        
        return encoder_hidden_states
    
    def load_pipeline(ckpt_path, device='cuda:0'):
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe
    
    
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "runwayml/stable-diffusion-v1-5"

    ##############################################################
    # ckpt_path = "/mnt/nas5/joonsung/2025/ckpts/sd-pokemon-checkpoint/sd-pokemon-checkpoint"
    
    # ckpt_path = 'runwayml/stable-diffusion-v1-5'
    

    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.ckpt_path, subfolder="tokenizer", revision=None
    # )
    # # tokenizer = tokenizer.to(device)
    # # tokenizer = tokenizer.cuda()

    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.ckpt_path, subfolder="text_encoder", revision=None
    # )
    # text_encoder = text_encoder.to(device)

    # vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    # vae = vae.to(device)

    # unet = UNet2DConditionModel.from_pretrained(
    #     args.ckpt_path, subfolder="unet", revision=None
    # )
    # unet = unet.to(device)
    
    # text_encoder.requires_grad_(False)
    
    # for p in text_encoder.parameters():
    #     p.requires_grad = False
    
    pipe = load_pipeline(args.ckpt_path, device)
    ##############################################################
    set_random_seed(args.gen_seed)

    # print(pipe.text_encoder.__class__)
    # print(pipe.text_encoder.config)
    # print(pipe.text_encoder.name_or_path)

    from transformers import CLIPTextModel
    from transformers.models.clip.modeling_clip import CLIPTextTransformer
    from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling


    class CLIPTextTransformerWithEmbeds(CLIPTextTransformer):
        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            inputs_embeds=None,
            **kwargs
        ):
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embeddings(
                    input_ids=input_ids,
                    position_ids=position_ids
                )

            encoder_outputs = self.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                **kwargs
            )
            last_hidden_state = encoder_outputs[0]
            last_hidden_state = self.final_layer_norm(last_hidden_state)

            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(torch.int).argmax(dim=-1) if input_ids is not None else 0
            ]

            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )

    # 2. 외부 CLIPTextModel patch

    class CLIPTextModelWithEmbeds(CLIPTextModel):
        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            inputs_embeds=None,
            **kwargs
        ):
            return self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                **kwargs
            )

    # 3. 모델 구성
    text_encoder_JS = CLIPTextModelWithEmbeds.from_pretrained(args.ckpt_path + "/text_encoder").to(device)

    # 내부 transformer 교체
    old_text_model = text_encoder_JS.text_model
    new_text_model = CLIPTextTransformerWithEmbeds(old_text_model.config).to(device)
    new_text_model.load_state_dict(old_text_model.state_dict())
    text_encoder_JS.text_model = new_text_model
    ##############################################################

    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    
    image = Image.open(args.img_path).convert("RGB")
    images = transform(image).unsqueeze(0).to(device)  # shape: (1, 3, 512, 512)
    
    prompt_id = args.img_path.split('/')[-1].split('.')[0]
    
    


        

    mse_loss = nn.MSELoss()
    
    garbage_prompt = ""
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[1].unsqueeze(0).detach()
    
    
    
    ###################################
    
    
    anchor = args.anchor
    until = args.until
    


    ## blip2
    if args.init == "ori":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/member_captions.jsonl"
        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/Dataset/SecMI_LDM_dataset/pokemon/non_member_captions.jsonl"
        else:
             ValueError("args.mem was not satisfied.")
             
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["filename"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )
    
        init = args.init
        
    
    ## ori - mem
    elif args.init == "clip_interrogator":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_members_caption_output.jsonl"

        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_Pokemon_non_members_caption_output.jsonl"
            
            
        with open(caption_path, "r") as f:
            prompt_to_caption = {
                json.loads(line)["prompt_id"]: json.loads(line)["caption"]
                for line in f
            }
            
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(pred_prompt)

        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )

        text_inputs = pipe.tokenizer(pred_prompt, padding='max_length', max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        # print(text_inputs)
        '''
        {'input_ids': tensor([[49406,   320,  2660,   705,   539,   320, 33082,  1061,  9528,  1069,
           593,   320,  1205,  3490,   267,  9054,  1012,   530,   787,  1710,
           267, 24584,  4353,   267, 39621,   267, 26051,   736,  5389,   267,
          2484,   267,  3010,   652,   267, 10724, 15039,  1710,   267,  7557,
          1518, 28315,   267, 14875,   267, 25233,   267,  3120,   835,  2134,
           267,  2103, 10648,   267, 35795,   267,  1767,  2159,   267, 12248,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0]])}
        '''
        token_ids = text_inputs.input_ids
        token_ids = token_ids.to(device)

        attention_mask = text_inputs.attention_mask
        attention_mask = attention_mask.to(device)

        if attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :] * attention_mask[:, None, :, None]
            attention_mask = attention_mask.to(dtype=init_text_embeddings.dtype)  # fp16 대응

        

        with torch.no_grad():
            token_embeds_init = pipe.text_encoder.get_input_embeddings()(token_ids)
        # print(token_embeds_init.shape) ## (1,77, 768)

        init = args.init

        
        
    else:
        ValueError("args.init was not satisfied.")

    
    # noise_scale = 0.1
    optim_iters = args.OptimIter
    iters = args.iters
    eps = args.eps
    step_size = eps/2.
    GUIDANCE_SCALE = 7.5
    # epsilon = 1e-5
    # step_size > 0.01: loss increase
    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
    
    # for p in pipe.text_encoder.parameters():
    #     p.requires_grad = False
    

    
    
    
    #### 1. Adv Example ####
    images = images*2. - 1.
    
    set_random_seed(args.gen_seed)
    
    
    
    
    #### 2. Text embedding ####
    ## New
    set_random_seed(args.gen_seed)
    
    # rnd_text_emb = init_text_embeddings[1].unsqueeze(0).detach() # (1, 77, 768)
    # optimized_token_embeds = token_embeds_init.clone().detach().requires_grad_(True)
    
    with torch.no_grad():
        latent = pipe.vae.encode(images)
        latent_z0 = 0.18215 * latent.latent_dist.sample()


    # rnd_text_emb.requires_grad_(True)
    # rnd_text_emb = rnd_text_emb.detach().clone().requires_grad_(True)
    optimized_token_embeds = token_embeds_init.clone().detach().requires_grad_(True)

    ### From Null-text inversion ###
    # optimizer = Adam([rnd_text_emb], lr=1e-2 * (1. - t / 100.))
    from torch.optim.lr_scheduler import CosineAnnealingLR

    lr = args.lr
    optimizer = Adam([optimized_token_embeds], lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=optim_iters, eta_min=1e-4)
    
    

    def get_noise_pred_single(pipe, latents, t, context):
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=context).sample
        # noise_pred = pipe.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    
        
    npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_1.npy"
    rnd_noise_np = np.load(npy_path)
    rnd_noise_2 = torch.from_numpy(rnd_noise_np).to(device)
    
    timesteps = list(range(0, 1000, 10))
    
    pbar = tqdm(range(iters))
    for it in pbar:
        
        rnd = args.emb_rnd
        t = torch.randint(anchor-(rnd//2), anchor+(rnd//2+1), (1,)).item()  # 0 ~ t 사이의 랜덤 정수
        
        ## anchor = 150 -> {110, 120, 130, 140, 150}

        # timesteps에서 해당 값 추출
        timestep = timesteps[t]
        timestep = torch.tensor([timestep], device=device)
        
        # rnd_noise_1 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
        # rnd_noise_2 = torch.randn(latent_z0.shape).to(device=device, dtype=latent_z0.dtype)
            
        # latent_zt_1 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        latent_zt_2 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_2.to(device), timestep)
        
        outputs = text_encoder_JS(input_ids=token_ids, inputs_embeds=optimized_token_embeds, attention_mask=attention_mask)
        text_embedding = outputs.last_hidden_state

        # noise_pred_uncond = get_noise_pred_single(pipe, latent_zt_1, timestep, uncond_emb.detach())
        noise_pred_cond = get_noise_pred_single(pipe, latent_zt_2, timestep, text_embedding)

        

        # loss_uncond = F.mse_loss(noise_pred_uncond.float(), rnd_noise_1.float().to(device), reduction="mean")
        loss_cond = F.mse_loss(noise_pred_cond.float(), rnd_noise_2.float().to(device), reduction="mean")
        
    
        # loss = (loss_cond - loss_uncond).to(pipe.device)
        
        loss = loss_cond.to(pipe.device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        
        ## Early Stopping ##
        # loss_item = loss.item()
        # if loss_item < epsilon + t * 2e-5:
        #     break
        
        if pbar is not None:
            pbar.set_description(
                # f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f} (cond: {loss_cond.item():.8f}, uncond: {loss_uncond.item():.8f})"
                f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.8f}"
            )
        
    # cond_embeddings_list.append(rnd_text_emb[:1].detach())/
    
    torch.cuda.empty_cache()
    gc.collect()
    
    with torch.no_grad():
        token_embedding_weight = pipe.text_encoder.get_input_embeddings().weight  # (49408, 768)
        token_embedding_weight = token_embedding_weight.detach()

        optimized_token_embeds = optimized_token_embeds.detach()  # (1, 77, 768)

        batch_size = 49408
        chunk_size = 1000  # GPU 메모리 상황에 따라 조절

        cosine_sims = []
        for i in range(0, batch_size, chunk_size):
            sub_token_embeds = token_embedding_weight[i:i+chunk_size]  # (chunk_size, 768)
            sub_token_embeds = sub_token_embeds.unsqueeze(0).unsqueeze(0)  # (1, 1, chunk_size, 768)

            sim = F.cosine_similarity(
                optimized_token_embeds.unsqueeze(2),  # (1, 77, 1, 768)
                sub_token_embeds,                    # (1, 1, chunk_size, 768)
                dim=-1
            )  # (1, 77, chunk_size)

            cosine_sims.append(sim)

        cosine_sim = torch.cat(cosine_sims, dim=-1)  # (1, 77, 49408)

    # 가장 유사한 토큰 ID 찾기
    optimized_token_ids = cosine_sim.argmax(dim=-1)  # (1, 77)

    # tokens = [pipe.tokenizer.convert_ids_to_tokens(idx.item()) for idx in optimized_token_ids[0]]
    # print(tokens)

    decoded_text = pipe.tokenizer.decode(optimized_token_ids[0], skip_special_tokens=True)
    print(decoded_text)

    


    optimized_token_ids = optimized_token_ids.detach().cpu()
    
    optimized_token_ids_npy = optimized_token_ids.numpy()    


    ## Save .npy
        
    
    # if args.mem == "member":
    #     save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/TokenID_{args.type}_anchor{anchor}_init_{init}_iters{iters}_lr{eps}/members/{prompt_id}"
    # elif args.mem == "non_member":
    #     save_dir = f"/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/ver{args.ver}/TokenID_{args.type}_anchor{anchor}_init_{init}_iters{iters}_lr{eps}/non_members/{prompt_id}"
    # else:
    #     ValueError("args.mem was not satisfied.")

    # os.makedirs(save_dir, exist_ok=True)
    # np.save(save_dir+f"/{prompt_id}_anchor{anchor}_rnd{rnd}_iter{iters}_{init}_Adam.npy", optimized_token_ids_npy)


    # output_path = save_dir+f"/{prompt_id}_optimized_caption.txt"
    # with open(output_path, "w", encoding="utf-8") as f:
    #     f.write(decoded_text)



def main_text_clid(args):
    
    
    def get_noise_pred_single(pipe, latents, t, context):
        # noise_pred = pipe.unet(latents, t, encoder_hidden_states=context)["sample"]
        noise_pred = pipe.unet(latents, t, context)["sample"]
        return noise_pred
    
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "runwayml/stable-diffusion-v1-5"

    
    
    # pipe = LocalStableDiffusionPipeline.from_pretrained(
    #         model_id,
    #         torch_dtype=torch.float32,
    #         safety_checker=None,
    #         requires_safety_checker=False,
    #     )
    
    
    pt_dir = "/mnt/nas5/joonsung/2025/ckpts/runwayml_SDv1_5/diffusers_0_18_2"
    os.makedirs(pt_dir, exist_ok=True)
    
    # torch.save(pipe, pt_dir+"/runwayml_SDv1_5.pt")
    pipe = torch.load(pt_dir+"/runwayml_SDv1_5.pt")
    
    
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


    
    

    pipe = pipe.to(device)
    
    set_random_seed(args.gen_seed)
    


    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])
    
    gt_root = "/mnt/nas5/joonsung/Dataset/LAION_mi/members"
    # gt_root = "/mnt/nas5/joonsung/Dataset/LAION_mi/non_members"

    dataset = ImageDataset(gt_root, transform=transform, max_images=1000)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    

        

    mse_loss = nn.MSELoss()
    
    garbage_prompt = ""
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[1].unsqueeze(0).detach()
    
    ################
    anchor = args.anchor
    until = args.until
    
    # num_images = 1200

    ## blip2
    # caption_path = "/mnt/nas5/joonsung/2025/VLM/blip_2/captions/blip2_members_caption_output.jsonl"
    # # caption_path = "/mnt/nas5/joonsung/2025/VLM/blip_2/captions/blip2_non_members_caption_output.jsonl"
    # with open(caption_path, "r") as f:
    #     prompt_to_caption = {
    #         json.loads(line)["prompt_id"]: json.loads(line)["caption"]
    #         for line in f
    #     }
    
    ## ori - mem
    caption_path = "/mnt/nas5/joonsung/Dataset/LAION_mi/laion_mi_members_metadata.jsonl"
    with open(caption_path, "r") as f:
        prompt_to_caption = {
            str(json.loads(line)["idx"]): json.loads(line)["caption"]
            for line in f
        }
        
    ## ori -nonmem
    # caption_path = "/mnt/nas5/joonsung/Dataset/LAION_mi/laion_mi_non_members_metadata.jsonl"
    # with open(caption_path, "r") as f:
    #     captions_by_line = [json.loads(line.strip())["caption"] for line in f]
    
    
        
    ################

    
    # noise_scale = 0.1
    iters = 2000
    # eps = 0.5
    # step_size = 0.25
    GUIDANCE_SCALE = 7.5
    epsilon = 1e-5
    # step_size > 0.01: loss increase
    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)

    for i, (prompt_ids, images) in tqdm(enumerate(dataloader)):
            

        # images = 2.0 * images - 1.0
        images = images.to(device)
        prompt_id = prompt_ids[0].split('.')[0]
        
        ### Prompt ###
        ## 1: blip2, ori-mem ##
        pred_prompt = prompt_to_caption.get(prompt_id, "")
        print(f"{prompt_id}: {pred_prompt}")
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )
        
        ## ori-nonmem ##
        # pred_prompt = captions_by_line[int(prompt_id)-1]
        # print(pred_prompt)
        # init_text_embeddings = pipe._encode_prompt(
        #     pred_prompt, device, 1, True, negative_prompt=""
        # )
        
        ## 2 ##
        # pred_prompt = "ata"
        # init_text_embeddings = pipe._encode_prompt(
        #     pred_prompt, device, 1, True, negative_prompt=""
        # )
        
        

        #########################################################
        ## img -> z0
        set_random_seed(args.gen_seed)
        latent = encode_image(pipe, images, dtype=torch.float32)
        
        
        rnd_noise = torch.randn(latent.shape).to(device=device, dtype=latent.dtype)
        
        ## 1. Uncond. DDIM inversion ##
        # set_random_seed(args.gen_seed)
        # with torch.no_grad():
        #     inverted_latents = invert(pipe, anchor, latent, garbage_prompt, device=device, guidance_scale=0, num_inference_steps=50)
            
        ## 2. add_noise ##
        # at the below FOR loop
        pipe.scheduler.set_timesteps(50, device=device)
            
   
        
        #########################################################
        ## Text Embedding
        # rnd_text_emb = torch.randn(garbage_text_embeddings[0].shape).to(device=device, dtype=garbage_text_embeddings.dtype).unsqueeze(0)
        
        
        # rnd_text_emb = init_text_embeddings[1].unsqueeze(0).detach() + (torch.rand(*uncond_emb.shape)*2*eps-eps).to(device=device, dtype=garbage_text_embeddings.dtype)
        rnd_text_emb = init_text_embeddings[1].unsqueeze(0).detach() # (1, 77, 768)
        # print(rnd_text_emb)
        
        cond_embeddings_list = []

        
        ## main_text ##
        # latent_cur = inverted_latents[-1].unsqueeze(0)

        #########################################################
        for t in range(anchor, until, -1):
            set_random_seed(args.gen_seed)
            rnd_text_emb.requires_grad_(True)
            
            ### From Null-text inversion ###
            # optimizer = Adam([rnd_text_emb], lr=0.01 * (1. - t / 100.)) ## default
            optimizer = Adam([rnd_text_emb], lr=0.01) # 0.001
            # optimizer = SGD([rnd_text_emb], lr=0.1) ## xxx 
            
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)
            
            # print(pipe.scheduler.timesteps)
            timestep = pipe.scheduler.timesteps[50-t]
            print(f"timestep: {timestep}")
            
            pbar = tqdm(range(iters))
            for it in pbar:
                rnd_noise = torch.randn(latent.shape).to(device=device, dtype=latent.dtype)
                ## main_text ##
                # latent_prev = inverted_latents[anchor-t-2].unsqueeze(0)
                
                ## main_text_clid ##
                latent_cur = pipe.scheduler.add_noise(latent.to(device), rnd_noise.to(device), timestep.to(device))

                # with torch.no_grad():
                #     noise_pred_uncond = get_noise_pred_single(pipe, latent_cur, timestep, uncond_emb)
                    
                
            
                
                
                
                noise_pred_cond = get_noise_pred_single(pipe, latent_cur, timestep, rnd_text_emb)

                # noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                
                # latents_prev_rec  = pipe.scheduler.step(
                #     noise_pred, timestep, latent_cur, **extra_step_kwargs, return_dict=False
                # )[0]

                ## Diffusion loss
                MSE_loss = (mse_loss(noise_pred_cond, rnd_noise)).to(pipe.device)
                
                ## KL divergence
                target_mean = torch.zeros_like(noise_pred_cond)
                target_std = torch.ones_like(noise_pred_cond)

                # kl_loss = torch.distributions.kl_divergence(
                #     torch.distributions.Normal(noise_pred_cond, torch.ones_like(noise_pred_cond)),  # predicted distribution
                #     torch.distributions.Normal(target_mean, target_std)  # true standard normal
                # ).mean()

                optimizer.zero_grad()
                # loss = MSE_loss + 0.1*kl_loss
                loss = MSE_loss
                loss.backward()
                optimizer.step()
                
                lr_scheduler.step()
                
                with torch.no_grad():
                    test_noise = torch.randn(latent.shape).to(device=device, dtype=latent.dtype)
                    latent_cur = pipe.scheduler.add_noise(latent.to(device), test_noise.to(device), timestep.to(device))
                    noise_pred_cond = get_noise_pred_single(pipe, latent_cur, timestep, rnd_text_emb)
                    
                    test_loss = (mse_loss(noise_pred_cond, test_noise))
                    
                ## Early Stopping ##
                # loss_item = loss.item()
                # if loss_item < epsilon + t * 2e-5:
                #     break
                
                if (it+1)%100 == 0:
                    print(f"Train loss: {loss}")
                    print(f"Test loss: {test_loss}")
                
                if pbar is not None:
                    pbar.set_description(
                        # f"Image {i+1}/{len(dataloader)} | Optimizing: t={timestep} - {anchor-t+1}/{anchor-until} - Iter {it} - MSE loss: {MSE_loss.item():.8f}, KL loss: {kl_loss.item():.8f} (test_loss: {test_loss:.8f})"
                        f"Image {i+1}/{len(dataloader)} | Optimizing: t={timestep} - {anchor-t+1}/{anchor-until} - Iter {it} - MSE loss: {MSE_loss.item():.8f} (test_loss: {test_loss:.8f})"
                    )
                
            cond_embeddings_list.append(rnd_text_emb[:1].detach())
            with torch.no_grad():
                latent_cur, _ = pipe.mtcnp(perturb_embeds=rnd_text_emb, perturb_latent=latent_cur, prompt=garbage_prompt, anchor=t, track_noise_norm=True)
            
        
            
        print(torch.max(rnd_text_emb))
        print(torch.min(rnd_text_emb))
        print(torch.mean(rnd_text_emb)) 
        rnd_text_emb_cpu = rnd_text_emb.detach().cpu().numpy()
        
        save_dir = f"/mnt/nas5/joonsung/2025/text_cond/LAION_mi_members/perturb_emb/SDv1_5_{anchor}to{until}_iters{iters}_init_ori"
        # save_dir = f"/mnt/nas5/joonsung/2025/text_cond/LAION_mi_non_members/perturb_emb/SDv1_5_{anchor}to{until}_iters{iters}_init_ori"
        
        os.makedirs(save_dir, exist_ok=True)
        np.save(save_dir+f"/{prompt_ids[0].split('.')[0]}_from{anchor}_to{until}_iter{iters}_blip2_Adam.npy", rnd_text_emb_cpu)
        
        if i+1 == 100:
            break
        


def main_latent(args):
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = args.model_id

    
    
    # pipe = LocalStableDiffusionPipeline.from_pretrained(
    #         args.model_id,
    #         torch_dtype=torch.float32,
    #         safety_checker=None,
    #         requires_safety_checker=False,
    #     )
    pt_dir = "/mnt/nas5/joonsung/2025/ckpts/runwayml_SDv1_5/diffusers_0_18_2"
    os.makedirs(pt_dir, exist_ok=True)
    
    # torch.save(pipe, pt_dir+"/runwayml_SDv1_5.pt")
    pipe = torch.load(pt_dir+"/runwayml_SDv1_5.pt")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


    pipe = pipe.to(device)
    
    set_random_seed(args.gen_seed)
    


    
    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])
    
    gt_root = "/mnt/nas5/joonsung/Dataset/LAION_mi/members"

    dataset = ImageDataset(gt_root, transform=transform, max_images=1000)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    

        

    loss = nn.MSELoss()
    
    garbage_prompt = "aka"
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[0].unsqueeze(0).detach()
    
    ################
    anchor = args.anchor
    until = args.until
    
    # num_images = 1200

    ## blip2
    # caption_path = "/mnt/nas5/joonsung/2025/VLM/blip_2/captions/blip2_members_caption_output.jsonl"
    # # caption_path = "/mnt/nas5/joonsung/2025/VLM/blip_2/captions/blip2_non_members_caption_output.jsonl"
    # with open(caption_path, "r") as f:
    #     prompt_to_caption = {
    #         json.loads(line)["prompt_id"]: json.loads(line)["caption"]
    #         for line in f
    #     }
    
    ## ori
    caption_path = "/mnt/nas5/joonsung/Dataset/LAION_mi/laion_mi_members_metadata.jsonl"
    with open(caption_path, "r") as f:
        prompt_to_caption = {
            str(json.loads(line)["idx"]): json.loads(line)["caption"]
            for line in f
        }
    ################

    
    noise_scale = 0.1
    iters = 100
    eps = 0.5
    step_size = 0.025
    
    # step_size > 0.01: loss increase
    


    for i, (prompt_ids, images) in tqdm(enumerate(dataloader)):
            
        # if i+1<1201:
        #     continue
        
        images = images.to(device)
        prompt_id = prompt_ids[0].split('.')[0]
        
        ### Prompt ###
        ## 1 ##
        # pred_prompt = prompt_to_caption.get(prompt_id, "")
        # print(pred_prompt)
        # init_text_embeddings = pipe._encode_prompt(
        #     pred_prompt, device, 1, True, negative_prompt=""
        # )
        
        ## 2 ##
        pred_prompt = "an image"
        init_text_embeddings = pipe._encode_prompt(
            pred_prompt, device, 1, True, negative_prompt=""
        )
        
        

        # if prompt_ids[0].split('.')[0] not in valid_prompt_ids:
        #     print(f"SSCD passing: {prompt_ids[0].split('.')[0]}")
        #     continue
        
        
        ## img -> z0
        set_random_seed(args.gen_seed)
        latent = encode_image(pipe, images, dtype=torch.float32)
        
        ## Uncond. DDIM inversion ##
        set_random_seed(args.gen_seed)
        with torch.no_grad():
            inverted_latent = invert(pipe, anchor, latent, garbage_prompt, device=device, guidance_scale=0, num_inference_steps=50)[-1][None]
            
        # ****** HERE ***** #

        adv_latent = inverted_latent.clone().detach() + (torch.rand(*inverted_latent.shape)*2*eps-eps).to(device=device, dtype=inverted_latent.dtype)

        trg_noise = torch.randn(inverted_latent.shape).to(device=device, dtype=torch.float32)
        
        
        # for p in pipe.unet.parameters():
        #     print(p.requires_grad) ## all True
            
        
        pbar = tqdm(range(iters))
        for it in pbar:


            actual_step_size = step_size - (step_size - step_size / 100) / iters * it
            # actual_step_size = step_size 
            
            

            
            for t in range(anchor, until, -1):

                set_random_seed(args.gen_seed)
                
                adv_latent.requires_grad_(True)
                
                _, noise_pred = pipe.mtcnp(perturb_embeds=None, perturb_latent=adv_latent, prompt=pred_prompt, anchor=t, track_noise_norm=True)


                # print(noise_pred_text.requires_grad) ## True
                # print()
                
                # pipe.unet.zero_grad()
                
                # cost = (rnd_noise-noise_pred_text).norm(p=2) / (anchor-until)
                cost = (loss(trg_noise, noise_pred) / (anchor-until)).to(pipe.device)
                # cost = (noise_pred_text).norm(p=2) / until
                
                # cost.backward()
                
                
                ## 1 ##
                # grad = rnd_text_emb.grad.detach()
                # grad = grad.sign()
                
                # rnd_text_emb = rnd_text_emb - actual_step_size*grad
                
                # rnd_text_emb = rnd_text_emb.detach()
                
                if pbar is not None:
                    pbar.set_description(
                        f"Image {i+1}/{len(dataloader)} | Iter {it} | Optimizing {anchor-t+1}/{anchor-until} | dtype: {adv_latent.dtype} - Step size: {actual_step_size:.4f} | Current loss: {cost.item():.6f}"
                    )
                
                ### 2 ###
                # rnd_text_emb = adv_latent - actual_step_size*adv_latent.grad.sign()
                # rnd_text_emb = rnd_text_emb.detach_()
                ## PGD (1)
                # perturb_latent = adv_latent - actual_step_size*adv_latent.grad.sign()
                # eta = torch.clamp(perturb_latent.data - inverted_latent.data, min=-eps, max=eps)
                # adv_latent = torch.clamp(perturb_latent + eta, min=0, max=1).detach_()
                
                ## PGD (2)
                grad, = torch.autograd.grad(cost, [adv_latent])
                adv_latent = adv_latent - grad.detach().sign() * actual_step_size
                adv_latent = torch.minimum(torch.maximum(adv_latent, adv_latent - eps), adv_latent + eps)
                adv_latent.grad = None
                torch.cuda.empty_cache()
      
            
            # rnd_text_emb.requires_grad_(True)
            # null_cost = loss(uncond_emb, rnd_text_emb)
            
            # null_cost.backward()
                
            # grad = rnd_text_emb.grad.detach()
            # grad = grad.sign()
            
            # rnd_text_emb = rnd_text_emb - actual_step_size*grad
            
            # rnd_text_emb = rnd_text_emb.detach()
            
            
        # rnd_text_emb_cpu = rnd_text_emb.detach().cpu().numpy()
        # np.save(f"/mnt/nas5/joonsung/2025/sd/IIP/noise_pred/LAION_mi_members/perturb_emb/{prompt_ids[0].split('.')[0]}_from{anchor}_to{until}_iter{iters}_step{step_size}_noisescale{noise_scale}.npy", rnd_text_emb_cpu)
        
        break
        


            
            
            
        
        
        # **************** #




def inference(args):
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = args.model_id

    pipe = LocalStableDiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )
    # pipe = torch.load("/mnt/nas5/joonsung/2025/ckpts/CompVis_SDv1_4/diffusers_0_18_2/CompVis_SDv1_4.pt")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # torch.save(pipe, "/mnt/nas5/joonsung/2025/ckpts/CompVis_SDv1_4/diffusers_0_18_2/CompVis_SDv1_4.pt")
    # pipe = torch.load("/mnt/nas5/joonsung/2025/ckpts/CompVis_SDv1_4/CompVis_SDv1_4.pt")
    pipe = pipe.to(device)
    
    set_random_seed(args.gen_seed)
    
    


    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),  # 이미지를 [0,1]로 정규화 후 CHW로 변환
    ])
    
    #################################
    anchor = args.anchor
    # anchor = None
    
    save_dir = "/mnt/nas5/joonsung/2025/text_cond/LAION_mi_members"
    # save_dir = "/mnt/nas5/joonsung/2025/text_cond/LAION_mi_non_members"
    
    emb_root = save_dir + "/perturb_emb"
    
    gt_root = "/mnt/nas5/joonsung/Dataset/LAION_mi/members"
    # gt_root = "/mnt/nas5/joonsung/Dataset/LAION_mi/non_members"
    dataset = ImageDataset(gt_root, transform=transform, max_images=None)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    
    

    if anchor != None:
        gen_root = save_dir + f"/recons_0to{args.anchor}"
    else:
        gen_root = save_dir + f"/recons_full"
    os.makedirs(gen_root, exist_ok=True)
    
    all_metrics = ["uncond_noise_norm", "text_noise_norm", "tcnp_noise_norm", "eps_uncond_norm", "eps_cond_norm"]
    all_tracks = []
    
    
    ################################
    
    
    garbage_prompt = "aka"
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    rnd_anchor = torch.randn((1,4,64,64)).to(device=device, dtype=torch.float32)

    
    for i, (prompt_ids, images) in tqdm(enumerate(dataloader)):      
        
        # idx_dir = gen_root + f"/{prompt_ids[0]}"
        # os.makedirs(idx_dir, exist_ok=True)
        
        
        # images = 2.0 * images - 1.0
        images = images.to(device)
        # if prompt_ids[0].split('.')[0] not in valid_prompt_ids:
        #     print(f"SSCD passing: {prompt_ids[0].split('.')[0]}")
        #     continue
        
        
        
        
        target_id = f"{prompt_ids[0].split('.')[0]}_"
        ## Load perturb_emb 
        matched_files = [
            f for f in os.listdir(emb_root)
            if f.startswith(target_id) and f.endswith(".npy")
        ]

        if len(matched_files) == 0 or len(matched_files) > 1:
            print(f"Missing .npy file for prompt_id: {prompt_ids[0].split('.')[0]}")
            continue
        
    

        
        
        
        npy_path = os.path.join(emb_root, matched_files[0])
        array = np.load(npy_path)
        
        perturb_emb = torch.from_numpy(array).to(device=device, dtype=garbage_text_embeddings.dtype)
        
        
        
        ## img -> z0
        set_random_seed(args.gen_seed)
        latent = encode_image(pipe, images, dtype=torch.float32)
        
        ## Uncond. DDIM inversion ##
        set_random_seed(args.gen_seed)
        with torch.no_grad():
            inverted_latent = invert(pipe, args.anchor, latent, garbage_prompt, device=device, guidance_scale=0, num_inference_steps=50)[-1][None]
            

            
            
        set_random_seed(args.gen_seed)     
        with torch.no_grad():      
            image, track_stats = pipe.sample_rest(perturb_emb, perturb_latent=inverted_latent, prompt=garbage_prompt, anchor=args.anchor, eps=rnd_anchor, track_noise_norm=True)   
        
        
        
        image.images[0].save(f"{gen_root}/{prompt_ids[0].split('.')[0]}_from_{1000-20*args.anchor}.png")
        
        uncond_noise_norm, text_noise_norm, tcnp_noise_norm, eps_uncond_norm, eps_cond_norm = (
            track_stats["uncond_noise_norm"],
            track_stats["text_noise_norm"],
            track_stats["tcnp_noise_norm"],
            track_stats["eps_uncond_norm"],
            track_stats["eps_cond_norm"]
        )

        curr_line = {}
        curr_line["index"] = prompt_ids[0].split('.')[0]
        curr_line["seed"] = args.gen_seed
        for metric_i in all_metrics:
            values = locals()[metric_i] ## locals(): 모든 변수들을 딕셔너리 형태로 반환하는 내장 함수
            curr_line[f"{metric_i}"] = values


        all_tracks.append(curr_line)
        
        if i+1 == 10:
            break
            
        
            
    if anchor != None:
        write_jsonlines(all_tracks, save_dir+f"/results_0to{args.anchor}.jsonl")
    else:
        write_jsonlines(all_tracks, save_dir+f"/results_full.jsonl")
        


def inversion_blip2(args):
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = args.model_id

    # pipe = LocalStableDiffusionPipeline.from_pretrained(
    #         args.model_id,
    #         torch_dtype=torch.float16,
    #         safety_checker=None,
    #         requires_safety_checker=False,
    #     )
    pipe = torch.load("/mnt/nas5/joonsung/2025/ckpts/CompVis_SDv1_4/diffusers_0_18_2/CompVis_SDv1_4.pt")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # torch.save(pipe, "/mnt/nas5/joonsung/2025/ckpts/CompVis_SDv1_4/diffusers_0_18_2/CompVis_SDv1_4.pt")
    # pipe = torch.load("/mnt/nas5/joonsung/2025/ckpts/CompVis_SDv1_4/CompVis_SDv1_4.pt")
    pipe = pipe.to(device)
    
    set_random_seed(args.gen_seed)
    


    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),  # 이미지를 [0,1]로 정규화 후 CHW로 변환
    ])
    
    ################################
    # anchor = args.anchor
    anchor = None
    ################################
    
    save_dir = "/mnt/nas5/joonsung/2025/VLM/blip_2/results_mem"
    # save_dir = "/mnt/nas5/joonsung/2025/VLM/blip_2/res/lts_nonmem"
    if anchor != None:
        gen_root = save_dir + f"/recons_0to{args.anchor}"
    else:
        gen_root = save_dir + f"/recons_full"
    os.makedirs(gen_root, exist_ok=True)
    
    
    all_metrics = ["uncond_noise_norm", "text_noise_norm", "tcnp_noise_norm"]
    all_tracks = []
    
    gt_root = "/mnt/nas5/joonsung/Dataset/LAION_mi/members"
    # gt_root = "/mnt/nas5/joonsung/Dataset/LAION_mi/non_members"
    dataset = ImageDataset(gt_root, transform=transform, max_images=1000)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    
    caption_path = "/mnt/nas5/joonsung/2025/VLM/blip_2/captions/blip2_members_caption_output.jsonl"
    # caption_path = "/mnt/nas5/joonsung/2025/VLM/blip_2/captions/blip2_non_members_caption_output.jsonl"
    with open(caption_path, "r") as f:
        prompt_to_caption = {
            json.loads(line)["prompt_id"]: json.loads(line)["caption"]
            for line in f
        }


    
    for i, (prompt_ids, images) in tqdm(enumerate(dataloader)):      
        
        # idx_dir = gen_root + f"/{prompt_ids[0]}"
        # os.makedirs(idx_dir, exist_ok=True)
        
        
        
        images = images.to(device)
        prompt_id = prompt_ids[0].split('.')[0]  # 확장자 제거
        prompt = prompt_to_caption.get(prompt_id, "")

        if prompt == "":
            print(f"Warning: prompt not found for {prompt_id}")
            continue  # 혹은 default prompt 지정

        ## img -> z0
        set_random_seed(args.gen_seed)
        latent = encode_image(pipe, images, dtype=torch.float16)
        
        
        ## Uncond. DDIM inversion ##
        set_random_seed(args.gen_seed)
        with torch.no_grad():
            inverted_latent = invert(pipe, anchor, latent, prompt, device=device, guidance_scale=0, num_inference_steps=50)[-1][None]
            

            
            
        set_random_seed(args.gen_seed)     
        with torch.no_grad():      
            image, track_stats = pipe.sample_rest(perturb_embeds=None, perturb_latent=inverted_latent, prompt=prompt, anchor=anchor, track_noise_norm=True)   
        
        
        
        image.images[0].save(f"{gen_root}/{prompt_ids[0].split('.')[0]}.png")
        
        uncond_noise_norm, text_noise_norm, tcnp_noise_norm = (
            track_stats["uncond_noise_norm"],
            track_stats["text_noise_norm"],
            track_stats["tcnp_noise_norm"]
        )

        curr_line = {}
        curr_line["index"] = prompt_ids[0].split('.')[0]
        curr_line["seed"] = args.gen_seed
        for metric_i in all_metrics:
            values = locals()[metric_i] ## locals(): 모든 변수들을 딕셔너리 형태로 반환하는 내장 함수
            curr_line[f"{metric_i}"] = values


        all_tracks.append(curr_line)
        
        break
        
        
        

            
        
            
    if anchor != None:
        write_jsonlines(all_tracks, save_dir+f"/results_0to{args.anchor}.jsonl")
    else:
        write_jsonlines(all_tracks, save_dir+f"/results_full.jsonl")
        



def inversion_clip_interr(args):
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = args.model_id

    # pipe = LocalStableDiffusionPipeline.from_pretrained(
    #         args.model_id,
    #         torch_dtype=torch.float16,
    #         safety_checker=None,
    #         requires_safety_checker=False,
    #     )
    pipe = torch.load("/mnt/nas5/joonsung/2025/ckpts/CompVis_SDv1_4/diffusers_0_18_2/CompVis_SDv1_4.pt")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # torch.save(pipe, "/mnt/nas5/joonsung/2025/ckpts/CompVis_SDv1_4/diffusers_0_18_2/CompVis_SDv1_4.pt")
    # pipe = torch.load("/mnt/nas5/joonsung/2025/ckpts/CompVis_SDv1_4/CompVis_SDv1_4.pt")
    pipe = pipe.to(device)
    
    set_random_seed(args.gen_seed)
    


    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),  # 이미지를 [0,1]로 정규화 후 CHW로 변환
    ])
    
    ################################
    anchor = args.anchor
    # anchor = None
    ################################
    
    save_dir = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/results_mem"
    # save_dir = "/mnt/nas5/joonsung/2025/VLM/blip_2/res/lts_nonmem"
    if anchor != None:
        gen_root = save_dir + f"/recons_0to{args.anchor}"
    else:
        gen_root = save_dir + f"/recons_full"
    os.makedirs(gen_root, exist_ok=True)
    
    
    all_metrics = ["uncond_noise_norm", "text_noise_norm", "tcnp_noise_norm"]
    all_tracks = []
    
    gt_root = "/mnt/nas5/joonsung/Dataset/LAION_mi/members"
    # gt_root = "/mnt/nas5/joonsung/Dataset/LAION_mi/non_members"
    dataset = ImageDataset(gt_root, transform=transform, max_images=1000)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    
    caption_path = "/mnt/nas5/joonsung/2025/VLM/clip_interrogator/captions/ClipInterrogator_members_caption_output.jsonl"
    # caption_path = "/mnt/nas5/joonsung/2025/VLM/blip_2/captions/blip2_non_members_caption_output.jsonl"
    with open(caption_path, "r") as f:
        prompt_to_caption = {
            json.loads(line)["prompt_id"]: json.loads(line)["caption"]
            for line in f
        }


    
    for i, (prompt_ids, images) in tqdm(enumerate(dataloader)):      
        
        # idx_dir = gen_root + f"/{prompt_ids[0]}"
        # os.makedirs(idx_dir, exist_ok=True)
        
        
        
        images = images.to(device)
        prompt_id = prompt_ids[0].split('.')[0]  # 확장자 제거
        prompt = prompt_to_caption.get(prompt_id, "")

        if prompt == "":
            print(f"Warning: prompt not found for {prompt_id}")
            continue  # 혹은 default prompt 지정

        ## img -> z0
        set_random_seed(args.gen_seed)
        latent = encode_image(pipe, images, dtype=torch.float16)
        
        
        ## Uncond. DDIM inversion ##
        set_random_seed(args.gen_seed)
        with torch.no_grad():
            inverted_latent = invert(pipe, anchor, latent, prompt, device=device, guidance_scale=0, num_inference_steps=50)[-1][None]
            

            
            
        set_random_seed(args.gen_seed)     
        with torch.no_grad():      
            image, track_stats = pipe.sample_rest(perturb_embeds=None, perturb_latent=inverted_latent, prompt=prompt, anchor=anchor, track_noise_norm=True)   
        
        
        
        image.images[0].save(f"{gen_root}/{prompt_ids[0].split('.')[0]}.png")
        
        uncond_noise_norm, text_noise_norm, tcnp_noise_norm = (
            track_stats["uncond_noise_norm"],
            track_stats["text_noise_norm"],
            track_stats["tcnp_noise_norm"]
        )

        curr_line = {}
        curr_line["index"] = prompt_ids[0].split('.')[0]
        curr_line["seed"] = args.gen_seed
        for metric_i in all_metrics:
            values = locals()[metric_i] ## locals(): 모든 변수들을 딕셔너리 형태로 반환하는 내장 함수
            curr_line[f"{metric_i}"] = values


        all_tracks.append(curr_line)
        
        break
        
        
        

            
        
            
    if anchor != None:
        write_jsonlines(all_tracks, save_dir+f"/results_0to{args.anchor}.jsonl")
    else:
        write_jsonlines(all_tracks, save_dir+f"/results_full.jsonl")


def inversion_ori_mem(args):
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = args.model_id

    # pipe = LocalStableDiffusionPipeline.from_pretrained(
    #         args.model_id,
    #         torch_dtype=torch.float16,
    #         safety_checker=None,
    #         requires_safety_checker=False,
    #     )
    pipe = torch.load("/mnt/nas5/joonsung/2025/ckpts/CompVis_SDv1_4/diffusers_0_18_2/CompVis_SDv1_4.pt")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # torch.save(pipe, "/mnt/nas5/joonsung/2025/ckpts/CompVis_SDv1_4/diffusers_0_18_2/CompVis_SDv1_4.pt")
    # pipe = torch.load("/mnt/nas5/joonsung/2025/ckpts/CompVis_SDv1_4/CompVis_SDv1_4.pt")
    pipe = pipe.to(device)
    
    set_random_seed(args.gen_seed)
    

    
    anchor = args.anchor
    # anchor = None
    
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),  # 이미지를 [0,1]로 정규화 후 CHW로 변환
    ])
    
    save_dir = "/mnt/nas5/joonsung/2025/VLM/ori/results_mem"
    
    if anchor != None:
        gen_root = save_dir + f"/recons_0to{args.anchor}"
    else:
        gen_root = save_dir + f"/recons_full"
    os.makedirs(gen_root, exist_ok=True)
    
    all_metrics = ["uncond_noise_norm", "text_noise_norm", "tcnp_noise_norm"]
    all_tracks = []
    
    gt_root = "/mnt/nas5/joonsung/Dataset/LAION_mi/members"
    dataset = ImageDataset(gt_root, transform=transform, max_images=1000)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    
    
    caption_path = "/mnt/nas5/joonsung/Dataset/LAION_mi/laion_mi_members_metadata.jsonl"
    with open(caption_path, "r") as f:
        prompt_to_caption = {
            str(json.loads(line)["idx"]): json.loads(line)["caption"]
            for line in f
        }

    
    
    
    
    for i, (prompt_ids, images) in tqdm(enumerate(dataloader)):      
        
        # idx_dir = gen_root + f"/{prompt_ids[0]}"
        # os.makedirs(idx_dir, exist_ok=True)
        
        
        
        images = images.to(device)
        prompt_id = prompt_ids[0].split('.')[0]  # 확장자 제거
        prompt = prompt_to_caption.get(prompt_id, "")
        
        # **
        if prompt_id != "10004853":
            continue

        if prompt == "":
            print(f"Warning: prompt not found for {prompt_id}")
            continue  # 혹은 default prompt 지정

        ## img -> z0
        set_random_seed(args.gen_seed)
        latent = encode_image(pipe, images, dtype=torch.float16)
        
        
        ## Uncond. DDIM inversion ##
        set_random_seed(args.gen_seed)
        with torch.no_grad():
            inverted_latent = invert(pipe, anchor, latent, prompt, device=device, guidance_scale=0, num_inference_steps=50)[-1][None]
        
        # **
        zT_img = latent_to_img(pipe, inverted_latent[0])
        zT_img.save("./Attn/dump/10004853_at20.png")

            
            
        set_random_seed(args.gen_seed)     
        with torch.no_grad():      
            image, track_stats = pipe.sample_rest(perturb_embeds=None, perturb_latent=inverted_latent, prompt=prompt, anchor=anchor, track_noise_norm=True)   
        
        
        
        image.images[0].save(f"{gen_root}/{prompt_ids[0].split('.')[0]}.png")
        
        uncond_noise_norm, text_noise_norm, tcnp_noise_norm= (
            track_stats["uncond_noise_norm"],
            track_stats["text_noise_norm"],
            track_stats["tcnp_noise_norm"]
        )

        curr_line = {}
        curr_line["index"] = prompt_ids[0].split('.')[0]
        curr_line["seed"] = args.gen_seed
        for metric_i in all_metrics:
            values = locals()[metric_i] ## locals(): 모든 변수들을 딕셔너리 형태로 반환하는 내장 함수
            curr_line[f"{metric_i}"] = values


        all_tracks.append(curr_line)
        
        
        
        
        

            
        
    # if anchor != None:
    #     write_jsonlines(all_tracks, save_dir+f"/results_0to{args.anchor}.jsonl")
    # else:
    #     write_jsonlines(all_tracks, save_dir+f"/results_full.jsonl")
            

  

def inversion_ori_nonmem(args):
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = args.model_id

    # pipe = LocalStableDiffusionPipeline.from_pretrained(
    #         args.model_id,
    #         torch_dtype=torch.float16,
    #         safety_checker=None,
    #         requires_safety_checker=False,
    #     )
    pipe = torch.load("/mnt/nas5/joonsung/2025/ckpts/CompVis_SDv1_4/diffusers_0_18_2/CompVis_SDv1_4.pt")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # torch.save(pipe, "/mnt/nas5/joonsung/2025/ckpts/CompVis_SDv1_4/diffusers_0_18_2/CompVis_SDv1_4.pt")
    # pipe = torch.load("/mnt/nas5/joonsung/2025/ckpts/CompVis_SDv1_4/CompVis_SDv1_4.pt")
    pipe = pipe.to(device)
    
    set_random_seed(args.gen_seed)
    

    
    anchor = args.anchor
    # anchor = None
    
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),  # 이미지를 [0,1]로 정규화 후 CHW로 변환
    ])
    
    save_dir = "/mnt/nas5/joonsung/2025/VLM/ori/results_nonmem"
    
    if anchor != None:
        gen_root = save_dir + f"/recons_0to{args.anchor}"
    else:
        gen_root = save_dir + f"/recons_full"
    os.makedirs(gen_root, exist_ok=True)
    
    all_metrics = ["uncond_noise_norm", "text_noise_norm", "tcnp_noise_norm"]
    all_tracks = []
    
    gt_root = "/mnt/nas5/joonsung/Dataset/LAION_mi/non_members"
    dataset = ImageDataset(gt_root, transform=transform, max_images=1000)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    
    
    caption_path = "/mnt/nas5/joonsung/Dataset/LAION_mi/laion_mi_non_members_metadata.jsonl"
    with open(caption_path, "r") as f:
        captions_by_line = [json.loads(line.strip())["caption"] for line in f]

    
    
    
    
    for i, (prompt_ids, images) in tqdm(enumerate(dataloader)):      
        
        # idx_dir = gen_root + f"/{prompt_ids[0]}"
        # os.makedirs(idx_dir, exist_ok=True)
        
        
        
        images = images.to(device)
        prompt_id = int(prompt_ids[0].split('.')[0])  # 확장자 제거
        
        if prompt_id >= len(captions_by_line):
            print(f"⚠️ Warning: prompt_id {prompt_id} out of range")
            continue

        prompt = captions_by_line[prompt_id-1]
        
        # 이후 prompt 사용
        print(f"[{prompt_id}] → {prompt}")
        
        ## img -> z0
        set_random_seed(args.gen_seed)
        latent = encode_image(pipe, images, dtype=torch.float16)
        
        
        ## Uncond. DDIM inversion ##
        set_random_seed(args.gen_seed)
        with torch.no_grad():
            inverted_latent = invert(pipe, anchor, latent, prompt, device=device, guidance_scale=0, num_inference_steps=50)[-1][None]
            

            
            
        set_random_seed(args.gen_seed)     
        with torch.no_grad():      
            image, track_stats = pipe.sample_rest(perturb_embeds=None, perturb_latent=inverted_latent, prompt=prompt, anchor=anchor, track_noise_norm=True)   
        
        
        
        image.images[0].save(f"{gen_root}/{prompt_ids[0].split('.')[0]}.png")
        
        uncond_noise_norm, text_noise_norm, tcnp_noise_norm= (
            track_stats["uncond_noise_norm"],
            track_stats["text_noise_norm"],
            track_stats["tcnp_noise_norm"]
        )

        curr_line = {}
        curr_line["index"] = prompt_ids[0].split('.')[0]
        curr_line["seed"] = args.gen_seed
        for metric_i in all_metrics:
            values = locals()[metric_i] ## locals(): 모든 변수들을 딕셔너리 형태로 반환하는 내장 함수
            curr_line[f"{metric_i}"] = values


        all_tracks.append(curr_line)
        
               
        


            
        
    if anchor != None:
        write_jsonlines(all_tracks, save_dir+f"/results_0to{args.anchor}.jsonl")
    else:
        write_jsonlines(all_tracks, save_dir+f"/results_full.jsonl")
            
  
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="diffusion memorization")
    parser.add_argument("--anchor", default=20, type=int)
    parser.add_argument("--until", default=10, type=int)
    parser.add_argument("--ckpt_path", default="CompVis/stable-diffusion-v1-4")

    parser.add_argument("--gen_seed", default=0, type=int)
    
    parser.add_argument("--OptimIter", default=100, type=int)
    parser.add_argument("--iters", default=50, type=int)
    parser.add_argument("--eps", default=0.05, type=float)
    parser.add_argument("--lr", default=5e-2, type=float)
    
    parser.add_argument("--img_path", type=str)
    
    parser.add_argument("--mem", type=str)
    parser.add_argument("--init", type=str)
    parser.add_argument("--type", type=str)
    parser.add_argument("--adv_rnd", default=5, type=int)
    parser.add_argument("--emb_rnd", default=3, type=int)
    
    
    
    parser.add_argument("--optim", default=False, type=bool)
    parser.add_argument("--optim_text", default=False, type=bool)
    parser.add_argument("--optim_adv_text", default=False, type=bool)
    parser.add_argument("--ver", default=False, type=int)
    parser.add_argument("--optim_latent", default=False, type=bool)
    
    parser.add_argument("--inf", default=False, type=bool)
    
    parser.add_argument("--inv_blip2", default=False, type=bool)
    parser.add_argument("--inv_clipinterr", default=False, type=bool)
    parser.add_argument("--inv_ori_mem", default=False, type=bool)
    parser.add_argument("--inv_ori_nonmem", default=False, type=bool)

    args = parser.parse_args()
    
    if args.optim:
        main_text_perimg(args)

        # main_latent(args)
    elif args.ver == 99:
        # main_text(args)
        main_text_emb_pool(args)
    elif args.ver == 1:
        main_adv_text_per_img_1(args)
    elif args.ver == 2:
        main_adv_text_per_img_2(args)
    elif args.ver == 3:
        main_adv_text_per_img_3(args)
    elif args.ver == 5:
        main_adv_text_per_img_5(args)
    elif args.ver == 7:
        main_adv_text_per_img_7(args)
    elif args.ver == 8:
        main_adv_text_per_img_8(args)
    elif args.ver == 9:
        main_adv_text_per_img_9(args)
    elif args.ver == 10:
        main_adv_text_per_img_10(args)
    elif args.ver == 1004:
        main_adv_text_per_img_10_earlystopping(args)
    elif args.ver == 100:
        main_adv_text_per_img_10_fromEmb(args)
    elif args.ver == 1000:
        main_adv_text_per_img_10_OnlyEmb(args)
    elif args.ver == 1001:
        main_adv_text_per_img_10_OnlyEmb_optimizer(args)
    elif args.ver == 1002:
        main_adv_text_per_img_10_Adam(args)
    elif args.ver == 11:
        main_rnd_text_11(args)
    elif args.ver == 12:
        main_adv_text_per_img_12(args)
    elif args.ver == 13:
        main_adv_text_per_img_13(args)
        
        
    elif args.ver == 20:
        main_clean_adv_text_10(args)
    elif args.ver == 21:
        main_rnd_adv_text_10(args)
        
    elif args.inf:
        inference(args)
    
    elif args.inv_blip2:
        inversion_blip2(args)
    elif args.inv_ori_mem:
        inversion_ori_mem(args)
    elif args.inv_ori_nonmem:
        inversion_ori_nonmem(args)
    elif args.inv_clipinterr:
        inversion_clip_interr(args)
    else:
        print("NOTHING TO DO")