import os, json, csv, random
import glob

import logging
from typing import Any, Mapping, Iterable, Union, List, Callable, Optional
import torch
import numpy as np
from tqdm.auto import tqdm


from PIL import Image
from torch.utils.data import Dataset, DataLoader

COMMON_EXTS = [".jpg", ".png", ".jpeg", ".webp", ".bmp"]

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
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

        self.valid_prompt_ids = set()
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                self.valid_prompt_ids.add(data["prompt_id"])

        for prompt_id in os.listdir(gt_root):
            if prompt_id in self.valid_prompt_ids:
                gt_img_path = os.path.join(gt_root, prompt_id, "0.png")
                if os.path.exists(gt_img_path):
                    self.valid_paths.append((prompt_id, gt_img_path))
                else:
                    print(f"Skipping missing GT: {prompt_id}")
            else:
                continue  

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


class ImageTextDatasetJSONL(Dataset):
    def __init__(
        self,
        image_root,
        caption_path,
        tokenizer,
        transform=None,
        image_size=(512, 512),
        max_images=None,
        is_train=True,
        caption_column="caption",
        max_length=77,
        exts=COMMON_EXTS,   
    ):
        self.image_root = image_root
        self.caption_path = caption_path
        self.tokenizer = tokenizer
        self.transform = transform
        self.image_size = image_size
        self.is_train = is_train
        self.caption_column = caption_column
        self.max_length = max_length
        self.exts = exts


        self.captions, self.valid = self._read_captions_jsonl_and_validate(image_root, caption_path, exts)
        

        if max_images:
            self.valid = self.valid[:max_images]

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        img_name = self.valid[idx] 
        img_path = os.path.join(self.image_root, img_name)

   
        img = Image.open(img_path).convert("RGB").resize(self.image_size)
        if self.transform:
            pixel_values = self.transform(img)
        else:
            arr = np.array(img).astype(np.float32) / 255.0
            pixel_values = torch.from_numpy(arr).permute(2, 0, 1)


        tokenized_input_ids = self.tokenize_captions({self.caption_column: self.captions[img_name]})

        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized_input_ids[0],
            "image_name": img_name,
        }

    def _read_captions_jsonl_and_validate(self, image_root, path, exts):
        """
        JSONL을 읽어 {실제파일명(확장자 포함) : [caption]} 딕셔너리와
        유효한 파일명 리스트를 반환.
        """
        caps = {}
        valid_files = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                base = str(d["filename"])
                cap = d[self.caption_column]

                # 1) 그대로 존재하면 그대로 사용
                direct_path = os.path.join(image_root, base)
                if os.path.exists(direct_path) and os.path.isfile(direct_path):
                    fname = base
                else:
                    # 2) 확장자 붙여가며 탐색
                    fname = None
                    for ext in exts:
                        p = os.path.join(image_root, base + ext)
                        if os.path.exists(p) and os.path.isfile(p):
                            fname = base + ext
                            break

                if fname is None:
                    # 이미지가 실제로 없으면 스킵
                    continue

                # 리스트 형태로 저장(학습/검증 공용 처리)
                caps.setdefault(fname, []).append(cap)
                valid_files.append(fname)

        # 정렬(재현성)
        valid_files = sorted(list(set(valid_files)))
        return caps, valid_files

    def tokenize_captions(self, examples):
        captions = []
        for caption in examples[self.caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if self.is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column {self.caption_column} should contain either strings or lists of strings."
                )
        inputs = self.tokenizer(
            captions,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids



class ImageDataset(Dataset):
    def __init__(self, image_root, transform=None, image_size=(512, 512), max_images=None):
        self.image_root = image_root
        self.transform = transform
        self.image_size = image_size
        self.valid_paths = []

        # 폴더 내의 모든 .jpg 파일을 수집
        all_files = sorted(os.listdir(image_root))
        for filename in all_files:
            if filename.endswith(".jpg"):
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



class ImageTextDataset(Dataset):
    def __init__(
        self,
        image_root,
        caption_path,
        tokenizer,
        transform=None,
        image_size=(512, 512),
        max_images=None,
        is_train=True,
        caption_column="caption",
        max_length=77,   # << 추가
    ):
        self.image_root = image_root
        self.caption_path = caption_path
        self.tokenizer = tokenizer
        self.transform = transform
        self.image_size = image_size
        self.is_train = is_train
        self.caption_column = caption_column
        self.max_length = max_length   # << 추가

        self.captions = self._read_captions(caption_path)
        self.valid = [fn for fn in sorted(os.listdir(image_root)) if fn in self.captions]
        if max_images:
            self.valid = self.valid[:max_images]

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        img_name = self.valid[idx]
        img_path = os.path.join(self.image_root, img_name)

        # 이미지 로드
        img = Image.open(img_path).convert("RGB").resize(self.image_size)
        if self.transform:
            pixel_values = self.transform(img)
        else:
            arr = np.array(img).astype(np.float32) / 255.0
            pixel_values = torch.from_numpy(arr).permute(2, 0, 1)

        # 캡션 토크나이즈
        tokenized_input_ids = self.tokenize_captions({self.caption_column: self.captions[img_name]})

        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized_input_ids[0],
            "image_name": img_name,
        }

    def _read_captions(self, path):
        # txt(csv) 포맷 처리
        caps = {}
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 0:  # 헤더 스킵
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    fn, cap = line.split(",", 1)
                except ValueError:
                    continue
                caps.setdefault(fn, []).append(cap)
        return caps

    def tokenize_captions(self, examples):
        captions = []
        for caption in examples[self.caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if self.is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{self.caption_column}` should contain either strings or lists of strings."
                )
        inputs = self.tokenizer(
            captions,
            max_length=self.max_length,  # << self.max_length 사용
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids


@torch.no_grad()
def encode_image(pipe, input_tsr, dtype=torch.float16):
    with torch.no_grad():
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

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)
        
        if anchor != None and i == anchor:
            break

    return torch.cat(intermediate_latents)