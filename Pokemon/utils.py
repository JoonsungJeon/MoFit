import os
import glob
import json
import logging
from typing import Any, Mapping, Iterable, Union, List, Callable, Optional
import torch
import numpy as np
import random
from tqdm.auto import tqdm


from PIL import Image
from torch.utils.data import Dataset, DataLoader


def read_jsonlines(filename: str) -> Iterable[Mapping[str, Any]]:
    """Yields an iterable of Python dicts after reading jsonlines from the input file."""
    file_size = os.path.getsize(filename)
    with open(filename) as fp:
        for line in tqdm(
            fp.readlines(), desc=f"Reading JSON lines from {filename}", unit="lines"
        ):
            try:
                example = json.loads(line)
                yield example
            except json.JSONDecodeError as ex:
                logging.error(f'Input text: "{line}"')
                logging.error(ex.args)
                raise ex

def load_jsonlines(filename: str) -> List[Mapping[str, Any]]:
    """Returns a list of Python dicts after reading jsonlines from the input file."""
    return list(read_jsonlines(filename))

def get_dataset(dataset_name, pipe=None):
    if "jsonl" in dataset_name:
        dataset = load_jsonlines(dataset_name)
        prompt_key = "caption"
    else: 
        raise NotImplementedError
    
    return dataset, prompt_key

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)
    
    ## Same Adv example? ##
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 또는 ":16:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    
def write_jsonlines(
    objs: Iterable[Mapping[str, Any]], filename: str, to_dict: Callable = lambda x: x
):
    """Writes a list of Python Mappings as jsonlines at the input file."""
    with open(filename, "w") as fp:
        for obj in tqdm(objs, desc=f"Writing JSON lines at {filename}"):
            fp.write(json.dumps(to_dict(obj)))
            fp.write("\n")



class WenDataset(Dataset):
    def __init__(self, gt_root, jsonl_path, transform=None, image_size=(512, 512)):
        self.gt_root = gt_root
        self.prompt_folders = sorted(os.listdir(gt_root))
        self.transform = transform
        self.image_size = image_size
        self.valid_paths = []

         # 1. jsonl 파일에서 유효한 prompt_id 로드
        self.valid_prompt_ids = set()
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                self.valid_prompt_ids.add(data["prompt_id"])

        # 2. 해당 prompt_id에 해당하는 이미지 경로만 저장
        for prompt_id in os.listdir(gt_root):
            if prompt_id in self.valid_prompt_ids:
                gt_img_path = os.path.join(gt_root, prompt_id, "0.png")
                if os.path.exists(gt_img_path):
                    self.valid_paths.append((prompt_id, gt_img_path))
                else:
                    print(f"Skipping missing GT: {prompt_id}")
            else:
                continue  # jsonl에 없는 prompt_id는 무시

    def __len__(self):
        return len(self.valid_paths)

    def __getitem__(self, idx):
        prompt_id, img_path = self.valid_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.image_size)

        if self.transform:
            img = self.transform(img)
        else:
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)  # HWC → CHW

        return prompt_id, img
    


class ImageDataset(Dataset):
    def __init__(self, image_root, transform=None, image_size=(512, 512), max_images=None):
        self.image_root = image_root
        self.transform = transform
        self.image_size = image_size
        self.valid_paths = []

        # 폴더 내의 모든 .jpg 파일을 수집
        all_files = sorted(os.listdir(image_root))
        for filename in all_files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(image_root, filename)
                self.valid_paths.append((filename, img_path))
                
                if max_images is not None and len(self.valid_paths) >= max_images:
                    print(f"Reached max_images limit: {max_images}")
                    break
            else:
                print(f"Skipping non-jpg file: {filename}")

    def __len__(self):
        return len(self.valid_paths)

    def __getitem__(self, idx):
        img_name, img_path = self.valid_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.image_size)

        if self.transform:
            img = self.transform(img)
        else:
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)  # HWC → CHW

        return img_name, img

@torch.no_grad()
def encode_image(pipe, input_tsr, dtype=torch.float16):
    with torch.no_grad():
        latent = pipe.vae.encode(input_tsr.to(dtype=dtype) * 2 - 1)
    latent = 0.18215 * latent.latent_dist.sample()
    
    return latent

def encode_image_grad(pipe, input_tsr, dtype=torch.float16):
    latent = pipe.vae.encode(input_tsr.to(dtype=dtype) * 2 - 1)
    latent = 0.18215 * latent.latent_dist.sample()
    
    return latent

@torch.no_grad()
def latent_to_img(pipe, inverted_latents):
    with torch.no_grad():
        inverted_img = pipe.decode_latents(inverted_latents.unsqueeze(0))
    inverted_img = pipe.numpy_to_pil(inverted_img)[0]
    
    return inverted_img
    

## Inversion
def invert(
    pipe,
    anchor,
    start_latents,
    prompt,
    device,
    guidance_scale=3.5,
    num_inference_steps=80,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
):

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = [latents]

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)

    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0) ##
    
    
    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]
        

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
        # print(torch.max(noise_pred_text))
        # print(torch.min(noise_pred_text))
        # print(torch.mean(noise_pred_text))
        
        # print(torch.max(noise_pred_uncond))
        # print(torch.min(noise_pred_uncond))
        # print(torch.mean(noise_pred_uncond))
        
        ####
        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]
        
        # print(f"current_t: {current_t}")
        # print(f"next_t: {next_t}")


        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        # latents = pipe.scheduler.reverse(noise_pred, t, timesteps[i+1], latents, **extra_step_kwargs).prev_sample ##
        ####
        
        # Store
        intermediate_latents.append(latents)
        
        if anchor != None and i == anchor:
            break

    return torch.cat(intermediate_latents)



def invert_js(
    pipe,
    anchor,
    start_latents,
    prompt,
    device,
    guidance_scale=3.5,
    num_inference_steps=80,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
):

    # Encode prompt
    text_embeddings = pipe._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = [latents]

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)

    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0) ##
    
    
    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]
        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, current_t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, current_t, encoder_hidden_states=text_embeddings).sample

        # # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
        # print(torch.max(noise_pred_text))
        # print(torch.min(noise_pred_text))
        # print(torch.mean(noise_pred_text))
        
        # print(torch.max(noise_pred_uncond))
        # print(torch.min(noise_pred_uncond))
        # print(torch.mean(noise_pred_uncond))
        
        ####
        
        
        # print(f"current_t: {current_t}")
        # print(f"next_t: {next_t}")


        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        # latents = pipe.scheduler.reverse(noise_pred, t, timesteps[i+1], latents, **extra_step_kwargs).prev_sample ##
        ####
        
        # Store
        intermediate_latents.append(latents)
        
        if anchor != None and i == anchor:
            break

    return torch.cat(intermediate_latents)