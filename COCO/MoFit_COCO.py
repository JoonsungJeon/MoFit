import torch
import random
from utils import *
import argparse
import os
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torch.optim.adam import Adam
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
import gc


from diffusers_0_18_2 import StableDiffusionPipeline
from diffusers_0_18_2 import DDIMScheduler



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
    
    def load_pipeline(ckpt_path, device='cuda'):
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        return pipe
    
    
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    
    pipe = load_pipeline(args.ckpt_path, device)
    set_random_seed(args.gen_seed)


    

    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5]),
    ])
    
    image = Image.open(args.img_path).convert("RGB")
    images = transform(image).unsqueeze(0).to(device) 
    
    prompt_id = args.img_path.split('/')[-1].split('.')[0]
    
    


        

    mse_loss = nn.MSELoss()
    
    garbage_prompt = ""
    garbage_text_embeddings = pipe._encode_prompt(
        garbage_prompt, device, 1, True, negative_prompt=""
    )
    
    uncond_emb = garbage_text_embeddings[1].unsqueeze(0).detach()
    
    
    
    
    
    anchor = args.anchor
    until = args.until
    
    ### Save Dir ###
    folder_dir = args.save_dir
    if args.mem == "member":
        img_dir = f"{folder_dir}/members/adv_img"
    elif args.mem == "non_member":
        img_dir = f"{folder_dir}/non_members/adv_img"
    else:
        ValueError("args.mem was not satisfied.")

    os.makedirs(img_dir, exist_ok=True)


    
    ## ori - mem
    if args.init == "blip2":
        if args.mem == "member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/blip_2/captions/blip2_COCO_train_2500_captions_0818.jsonl"

        elif args.mem == "non_member":
            caption_path = "/mnt/nas5/joonsung/2025/VLM/blip_2/captions/blip2_COCO_test_2458_captions_0818.jsonl"
            
            
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

        
        
    else:
        ValueError("args.init was not satisfied.")

    

    optim_iters = args.OptimIter
    iters = args.iters
    eps = args.eps
    if eps == 0:
        step_size = 0.15
    else:
        step_size = args.step_size


    
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
    


    
    
    
    #### 1. Surrogate ####
    set_random_seed(args.gen_seed)
    

    
    images = images*2. - 1.
    
    
    gen = torch.Generator(device=device).manual_seed(args.gen_seed)
    
    if eps == 0:
        adv_img = images.clone().detach()
    else:
        init_noise = (torch.rand(*images.shape, generator=gen, device=device, dtype=torch.float32)*2*eps - eps)
        adv_img = images.clone().detach() + init_noise
        

    

    
    timesteps = list(range(0, 1000, 10))

    pbar_adv = tqdm(range(optim_iters))

    ## **** ##
    npy_path = "/mnt/nas5/joonsung/2025/adv_ex_emb2_Pokemon/rnd_noise_1.npy"
    rnd_noise_np = np.load(npy_path)
    rnd_noise_1 = torch.from_numpy(rnd_noise_np).to(device=device, dtype=adv_img.dtype)
    
    
    for j in range(anchor, until, -1):
        set_random_seed(args.gen_seed)


        for it in pbar_adv:

            adv_img = adv_img.detach().clone().requires_grad_(True)
            
            

            rnd = args.adv_rnd
            t = torch.randint(j-(rnd//2), j + (rnd//2+1), (1,)).item()  
            


            timestep = timesteps[t]
            timestep = torch.tensor([timestep], device=device)



            pipe.unet.zero_grad()
            pipe.vae.zero_grad()
            
            actual_step_size = step_size - (step_size - step_size / 100) / optim_iters * it

            
            adv_latent_x0 = pipe.vae.encode(adv_img.to(dtype=torch.float32)).latent_dist
            adv_latent_x0 = 0.18215 * adv_latent_x0.mean

            
            
            adv_latent_xt = pipe.scheduler.add_noise(adv_latent_x0.to(device), rnd_noise_1.to(device), timestep)


            _, _, noise_pred_uncond, noise_pred_text = pipe.mtcnp_adv(perturb_embeds=None, perturb_latent=adv_latent_xt, prompt=pred_prompt, anchor=t, guidance_scale=7.5) ## default: 7.5



             

            ## [DEFAULT] 1. Uncond ~ rnd noise ##
            if args.type == "Uncond":
                cost = mse_loss(noise_pred_uncond, rnd_noise_1)
                grad, = torch.autograd.grad(cost, [adv_img])
                adv_img = adv_img - grad.sign() * actual_step_size
                
            ## 2. Cond ~ rnd noise ##
            elif args.type == "Cond":
                cost = mse_loss(noise_pred_text, rnd_noise_1)
                grad, = torch.autograd.grad(cost, [adv_img])
                adv_img = adv_img - grad.sign() * actual_step_size
            
            

            adv_img.data = torch.clamp(adv_img, min=-1, max=1)
            adv_img.grad = None
            
            

            if pbar_adv is not None:
                pbar_adv.set_description(
                    f"Image: {prompt_id} | timestep {timestep.item()} | Iter {it} | eps {eps} --> Step size: {actual_step_size:.4f} / Current loss: {cost.item():.6f}"
                )

            
            

            
            

            

  
    

    torch.cuda.empty_cache()
    gc.collect()
    
    adv_img = adv_img.detach()
    

    adv_img_cpu = adv_img.cpu().squeeze(0)  # shape: (3, H, W)
    adv_img_cpu = (adv_img_cpu + 1) / 2  # Map to [0, 1]
    adv_img_pil = to_pil_image(adv_img_cpu)

    
    save_path = os.path.join(img_dir, f"{prompt_id}_anchor{anchor}_rnd{rnd}_OptimIter{optim_iters}_eps{eps}_step{step_size}.png")
    adv_img_pil.save(save_path)
    
    
    
    
    
    #### 2. Text embedding ####
    set_random_seed(args.gen_seed)
    
    
    init_text_emb = init_text_embeddings[1].unsqueeze(0).detach() # (1, 77, 768)
    
    with torch.no_grad():
        latent = pipe.vae.encode(adv_img)
        latent_z0 = 0.18215 * latent.latent_dist.sample()
        
        cln_latent = pipe.vae.encode(images)
        cln_latent_z0 = 0.18215 * cln_latent.latent_dist.sample()



    rnd_text_emb = init_text_emb.detach().clone().requires_grad_(True)
    optimizer = Adam([rnd_text_emb], lr=args.lr)
    
    

    def get_noise_pred_single(pipe, latents, t, context):
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=context).sample
        return noise_pred
    
    
        
    iters_list = list(range(100, iters + 1, 100))
    
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

    

    
    pbar = tqdm(range(iters))
    for it in pbar:
        
        rnd = args.emb_rnd
        t = torch.randint(j-(rnd//2), j+(rnd//2+1), (1,)).item()  
        

        timestep = timesteps[t]
        timestep = torch.tensor([timestep], device=device)
        

        latent_zt_2 = pipe.scheduler.add_noise(latent_z0.to(device), rnd_noise_1.to(device), timestep)
        
        noise_pred_cond = get_noise_pred_single(pipe, latent_zt_2, timestep, rnd_text_emb)

        loss_cond = F.mse_loss(noise_pred_cond.float(), rnd_noise_1.float().to(device), reduction="mean")
        
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

            cln_vlm_loss_cond = F.mse_loss(get_noise_pred_single(pipe, cln_latent_zt, timestep, init_text_emb).float(), rnd_noise_1.float().to(device), reduction="mean").to(device)
            
            eval_loss = (cln_loss_cond - cln_loss_uncond).to(pipe.device)
            eval_VLM_loss = (cln_vlm_loss_cond - cln_loss_uncond).to(pipe.device)
        
        
        
        if pbar is not None:
            pbar.set_description(
                f"Image:{prompt_id} | Optimizing: t={timestep.item()} | Iter {it} - Current loss: {loss.item():.6f} ||| inf-emb-cond: {cln_loss_cond.item():.6f} ||| inf-uncond: {cln_loss_uncond.item():.6f} ||| MoFit: {eval_loss.item():.6f} ||| VLM_MoFit: {eval_VLM_loss.item():.6f}"
            )

        if (it+1) in iters_list:
            rnd_text_emb_npy = rnd_text_emb.detach().cpu().numpy()
            save_it_dir = pre_dir + f"/perturb_emb_iter{it+1}_lr{args.lr}"
            np.save(save_it_dir+f"/{prompt_id}_anchor{anchor}_rnd{rnd}_iter{iters}_{init}_Adam.npy", rnd_text_emb_npy)
                
 


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="diffusion memorization")
    parser.add_argument("--anchor", default=20, type=int)
    parser.add_argument("--until", default=10, type=int)
    parser.add_argument("--ckpt_path", default="CompVis/stable-diffusion-v1-4")

    parser.add_argument("--gen_seed", default=0, type=int)
    
    parser.add_argument("--OptimIter", default=100, type=int)
    parser.add_argument("--iters", default=50, type=int)
    parser.add_argument("--eps", default=0.05, type=float)
    parser.add_argument("--step_size", default=0.15, type=float)
    parser.add_argument("--lr", default=5e-2, type=float)
    parser.add_argument("--thres", default=0.0002, type=float)
    
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--save_dir", type=str)
    
    parser.add_argument("--mem", type=str)
    parser.add_argument("--init", type=str)
    parser.add_argument("--type", type=str)
    parser.add_argument("--adv_rnd", default=5, type=int)
    parser.add_argument("--emb_rnd", default=3, type=int)
    
    


    args = parser.parse_args()

    main_adv_text_per_img_10(args)
    
    