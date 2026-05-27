import torchvision.transforms as transforms
from PIL import Image

import numpy as np
import torch
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel,  DDIMScheduler
from tqdm.auto import tqdm
import time
import torch.nn.functional as F

from datasets import Dataset as HFDataset, Features, Value, Image as HFImage, Array3D, Array2D



from utils import *
import json
import os



timestart = time.strftime('%m%d_%H%M%S', time.localtime()).split()[0]
device = "cuda" if torch.cuda.is_available() else "cpu"
print('device:', device)

class Flag(object):
    pass


#####################################################

Use_data_model_name = "COCO_blip" 
## LAION_blip, Laion_emb
## Pokemon_clip, Pokemon_emb
## COCO_blip, COCO_emb
## flickr_8k_blip, flickr_8k_emb

#####################################################

flags = Flag
ver = "COCO"
diff_path = {
   "v1_4":"CompVis/stable-diffusion-v1-4",
   "v1_5":"runwayml/stable-diffusion-v1-5",
   "v1_5_local":"path/to/ckpts/runwayml-SDv15",
   "Pokemon": "path/to/ckpts/sd-pokemon-checkpoint",
   "COCO": "path/to/ckpts/sd-MSCOCO-checkpoint",
   "flickr_8k": "path/to/ckpts/sd-flickr-checkpoint",
   
}[ver]


flags.diff_path = diff_path
flags.anchor = 0
print(f"Model: {diff_path}")


train_data_dict = {
    "LAION_blip": "members",
    "LAION_emb": "members",
    
    "Pokemon_clip": "members",
    "Pokemon_emb": "members",
    
    "COCO_blip": "members",
    "COCO_emb": "members",
    
    "flickr_8k_blip": "members",
    "flickr_8k_emb": "members",

}

if 'coco' in Use_data_model_name:
    if 'ori' in Use_data_model_name:
        flags.dataset_train_name = train_data_dict['coco_ori']
    elif 'split1' in Use_data_model_name:
        flags.dataset_train_name = train_data_dict['coco_split1']

else:
    flags.dataset_train_name = train_data_dict[Use_data_model_name]

test_data_dict = {
    "LAION_blip": "nonmembers",
    "LAION_emb": "nonmembers",
    
    "Pokemon_clip": "nonmembers",
    "Pokemon_emb": "nonmembers",
    
    "COCO_blip": "nonmembers",
    "COCO_emb": "nonmembers",
    
    "flickr_8k_blip": "nonmembers",
    "flickr_8k_emb": "nonmembers",
}

if 'coco' in Use_data_model_name:
    if 'ori__' in Use_data_model_name:
        flags.dataset_test_name = test_data_dict['coco_ori']
    elif 'split1' in Use_data_model_name:
        flags.dataset_test_name = test_data_dict['coco_split1']

else:
    flags.dataset_test_name = test_data_dict[Use_data_model_name]





#********************************************#
token = "write_down_your_own_token"
#********************************************#

### LOAD MODEL ###
vae = AutoencoderKL.from_pretrained(
    diff_path, subfolder='vae', use_auth_token=token)
print('vae loaded.')


tokenizer = CLIPTokenizer.from_pretrained(diff_path, subfolder="tokenizer", )
text_encoder = CLIPTextModel.from_pretrained(diff_path, subfolder="text_encoder", )
print('tokenizer, textencoder loaded.')

unet = UNet2DConditionModel.from_pretrained(
    diff_path,
    subfolder='unet', )  
print('unet loaded.')

scheduler = DDIMScheduler.from_pretrained(diff_path, subfolder="scheduler")
print('sch loaded.')



vae = vae.to(device)
text_encoder = text_encoder.to(device)
unet = unet.to(device)

vae.eval()
unet.eval()

flags.attack = 'MoFit'


flags.T = 1000
flags.even_num = 10  ## default = 10
flags.max_n_samples = 3  ##
flags.max_clid_samples = 3  ##

flags.trials_eacht = 1

flags.train_batch_size = 8
flags.dataloader_num_workers = 0
flags.resolution = 512
flags.image_column = "image"
flags.caption_column = "text"
flags.t_sec = 100
flags.timestep = 10
flags.stpsnumi = 1


    

if Use_data_model_name in ["LAION_GT", "LAION_blip", "LAION_emb"]:
    flags.outdir = 'path/to/MoFit/Results/LAION'    

if Use_data_model_name in ["Pokemon_GT", "Pokemon_clip", "Pokemon_emb"]:
    flags.outdir = 'path/to/MoFit/Results/Pokemon'

if Use_data_model_name in ["COCO_GT", "COCO_blip", "COCO_emb"]:
    flags.outdir = 'path/to/MoFit/Results/COCO'

if Use_data_model_name in ["flickr_8k_GT", "flickr_8k_blip", "flickr_8k_emb"]:
    flags.outdir = 'path/to/MoFit/Results/Flickr'
    
os.makedirs(flags.outdir, exist_ok=True)




Template_name = Use_data_model_name
Time = timestart

print(str(flags.__dict__) + '\n' + diff_path + '\n' + flags.attack + '\n' + Template_name + '-------' + '\n')

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)




def get_data_emb(flags=flags, Use_data_model_name=None, dataset_name=None, max_samples=1000):
    '''
    Loading data
    '''
    assert dataset_name != None

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        text_embeddings = torch.stack([torch.tensor(example["text_embedding"]) for example in examples])
        input_ids_null = torch.stack([example["input_ids_null"] for example in examples])
        return {
            "pixel_values": pixel_values,
            "text_embedding": text_embeddings,
            "input_ids_null": input_ids_null
        }
    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(flags.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(flags.resolution),  # if args.center_crop else transforms.RandomCrop(args.resolution),
            # transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    flags.dataset_config_name = None
    flags.cache_dir = None
    flags.train_data_dir = None
    import os

    print("Loading Data...")
    
    if Use_data_model_name == "LAION_emb":
        if dataset_name == "members":
            image_dir = "path/to/LAION_mi/members/clean/images"
            embedding_dir = "path/to/LAION_mi/members/emb"
            
        elif dataset_name == "nonmembers":
            image_dir = "path/to/LAION_mi/non_members/clean/images"            
            embedding_dir = "path/to/LAION_mi/non_members/emb"
        else:
            raise ValueError(f"{dataset_name} is an invalid dataset_name.")
        
    if Use_data_model_name == "Pokemon_emb":


        if dataset_name == "members":
            image_dir = "path/to/Pokemon/members/clean/images"
            embedding_dir = "path/to/Pokemon/members/emb"
            
            
            
        elif dataset_name == "nonmembers":
            image_dir = "path/to/Pokemon/non_members/clean/images"         
            embedding_dir = "path/to/Pokemon/non_members/emb"
            
        else:
            raise ValueError(f"{dataset_name} is an invalid dataset_name.")
    

    if Use_data_model_name == "flickr_8k_emb":

        if dataset_name == "members":
            image_dir = "path/to/flickr/members/clean/images"           
            embedding_dir = "path/to/flickr/members/emb"
            
            
            
        elif dataset_name == "nonmembers":
            image_dir = "path/to/flickr/non_members/clean/images"         
            embedding_dir = "path/to/flickr/non_members/emb"
            
        else:
            raise ValueError(f"{dataset_name} is an invalid dataset_name.")
        
        
    if Use_data_model_name == "COCO_emb":
        emb_iter = 300
        lr = 0.06
        
        if dataset_name == "members":
            image_dir = "path/to/MSCOCO/members/clean/images"
            embedding_dir = f"path/to/Results/COCO/members/members/perturb_emb_iter{emb_iter}_lr{lr}"
            
            
            flags.anchor = anchor
            flags.lr = lr
            
        elif dataset_name == "nonmembers":
            image_dir = "path/to/MSCOCO/non_members/clean/images"         
            embedding_dir = f"path/to/Results/COCO/members/members/perturb_emb_iter{emb_iter}_lr{lr}"

            flags.anchor = anchor
            flags.lr = lr
        else:
            raise ValueError(f"{dataset_name} is an invalid dataset_name.")
    

    images, text_embeddings = [], []
    count = 0

    for fname in sorted(os.listdir(image_dir)):
        if not (fname.endswith(".jpg") or fname.endswith(".png")):
            continue

        idx = os.path.splitext(fname)[0]
        img_path = os.path.join(image_dir, fname)

        matching_files = [f for f in os.listdir(embedding_dir) if f.startswith(f"{idx}_") and f.endswith(".npy")]
        if len(matching_files) == 0:
            print("NO embeddings")
            continue
        

        embedding_path = os.path.join(embedding_dir, matching_files[0])

        try:
            Image.open(img_path).verify()
            embedding = np.load(embedding_path)

            images.append({"path": img_path})
            text_embeddings.append(embedding)



            count += 1
            if max_samples > 0 and count >= max_samples:
                break
        except Exception as e:
            print(f"Error loading {idx}: {e}")

    if len(text_embeddings) == 0:
        raise RuntimeError("No embeddings found. Please check your paths and file structure.")


    features = Features({
        "image": HFImage(),
        "text_embedding": Array3D(dtype="float32", shape=text_embeddings[0].shape),
        # "inputs_null": Array2D(dtype="int64", shape=(inputs_nulls[0].shape[0],)),  # usually (77,)
    })

    dataset = HFDataset.from_dict(
        {"image": images, "text_embedding": text_embeddings},
        features=features
    )

    def preprocess_train_multi(examples):
        images = [image.convert("RGB") for image in examples["image"]]
        
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids_null"] = tokenizer([""], padding="max_length", truncation=True, return_tensors="pt").input_ids
        

        
        return examples

    dataset = dataset.with_transform(preprocess_train_multi)


    
    from torch.utils.data import Subset
    if max_samples > 0:
        subset_indices = range(min(len(dataset), max_samples))
    else:
        subset_indices = range(len(dataset))
    subset_dataset = Subset(dataset, subset_indices)

    test_dataloader = torch.utils.data.DataLoader(
        subset_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=flags.train_batch_size,
        num_workers=flags.dataloader_num_workers,
    )

    print(f"{len(subset_dataset)} (image, embedding, inputs_null) triples loaded.")
    return test_dataloader







def get_data(flags=flags, Use_data_model_name=None, dataset_name=None, max_samples=1000):
    '''
    Loading data
    '''
    assert dataset_name != None

    # DataLoaders creation:
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        input_ids_1 = torch.stack([example["input_ids_1"] for example in examples])
        input_ids_2 = torch.stack([example["input_ids_2"] for example in examples])
        input_ids_3 = torch.stack([example["input_ids_3"] for example in examples])
        input_ids_null = torch.stack([example["input_ids_null"] for example in examples])

        return {"pixel_values": pixel_values, "input_ids": input_ids, "input_ids_1": input_ids_1,
                "input_ids_2": input_ids_2, "input_ids_3": input_ids_3, "input_ids_null": input_ids_null, }

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(flags.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(flags.resolution),  # if args.center_crop else transforms.RandomCrop(args.resolution),
            # transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    flags.dataset_config_name = None
    flags.cache_dir = None
    flags.train_data_dir = None
    import os

    print("Loading Data...")
    
    
    if Use_data_model_name == "LAION_blip":
        if dataset_name == "members":
            image_dir = "path/to/LAION_mi/members/clean/images"
            jsonl_path = "path/to/data/Captions/LAION_mi/blip2_members_1000.jsonl"

            # 1. jsonl load → dict: {idx: caption}
            idx_to_caption = {}
            with open(jsonl_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    idx_to_caption[str(data["prompt_id"])] = data["caption"]
                    
        elif dataset_name == "nonmembers":
            image_dir = "path/to/LAION_mi/non_members/clean/images"
            jsonl_path = "path/to/data/Captions/LAION_mi/blip2_non_members_1000.jsonl"

            # 1. jsonl load → dict: {idx: caption}
            idx_to_caption = {}
            with open(jsonl_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    idx_to_caption[str(data["prompt_id"])] = data["caption"]
                    
        else:
            raise ValueError(f"{dataset_name} is an invalid dataset_name.")
        
    

    
        
    if Use_data_model_name == "Pokemon_clip":
        if dataset_name == "members":
            image_dir = "path/to/Pokemon/members/clean/images"
            

            jsonl_path = "path/to/data/Captions/Pokemon/ClipInterrogator_Pokemon_members_captions.jsonl"

            # 1. jsonl load → dict: {idx: caption}
            idx_to_caption = {}
            with open(jsonl_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    idx_to_caption[str(data["prompt_id"])] = data["caption"]
                    
        elif dataset_name == "nonmembers":
            image_dir = "path/to/Pokemon/non_members/clean/images"
            

            jsonl_path = "path/to/data/Captions/Pokemon/ClipInterrogator_Pokemon_non_members_captions.jsonl"

            # 1. jsonl load → dict: {idx: caption}
            idx_to_caption = {}
            with open(jsonl_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    idx_to_caption[str(data["prompt_id"])] = data["caption"]
                    
        else:
            raise ValueError(f"{dataset_name} is an invalid dataset_name.")
    
    
    
    
    if Use_data_model_name == "COCO_blip":
        if dataset_name == "members":
            image_dir = "path/to/MSCOCO/members/clean/images"
            jsonl_path = "path/to/data/Captions/COCO/blip2_members_2500_captions.jsonl"


            # 1. Load jsonl
            idx_to_caption = {}
            with open(jsonl_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    idx_to_caption[str(data["filename"])] = data["caption"]
                    
        elif dataset_name == "nonmembers":
            image_dir = "path/to/MSCOCO/non_members/clean/images"
            jsonl_path = "path/to/data/Captions/COCO/blip2_non_members_2458_captions.jsonl"



            # 1. Load jsonl
            idx_to_caption = {}
            with open(jsonl_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    idx_to_caption[str(data["filename"])] = data["caption"]
                    
        else:
            raise ValueError(f"{dataset_name} is an invalid dataset_name.")
    
    
        
    if Use_data_model_name == "flickr_8k_blip":
        if dataset_name == "members":
            image_dir = "path/to/Flickr/members/clean/images"
            
            jsonl_path = "path/to/data/Captions/Flickr/blip2_member_2500_captions.jsonl"

            # 1. jsonl load → dict: {idx: caption}
            idx_to_caption = {}
            with open(jsonl_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    idx_to_caption[str(data["prompt_id"])] = data["caption"]
                    
        elif dataset_name == "nonmembers":
            image_dir = "path/to/Flickr/non_members/clean/images"
            
            jsonl_path = "path/to/data/Captions/Flickr/blip2_non_member_2500_captions.jsonl"

            # 1. jsonl load → dict: {idx: caption}
            idx_to_caption = {}
            with open(jsonl_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    idx_to_caption[str(data["prompt_id"])] = data["caption"]
                    
        else:
            raise ValueError(f"{dataset_name} is an invalid dataset_name.")
    
    
        
        
    
    images, captions = [], []
    count = 0

    for fname in sorted(os.listdir(image_dir)):
        if not (fname.endswith(".jpg") or fname.endswith(".png")):
            continue

        idx = os.path.splitext(fname)[0]
        img_path = os.path.join(image_dir, fname)

        if idx not in idx_to_caption:
            continue

        if not os.path.exists(img_path):
            continue

        try:
            Image.open(img_path).verify()  
            images.append({"path": img_path})  
            captions.append(idx_to_caption[idx])
            count += 1
            if max_samples > 0 and count >= max_samples:
                break
        except Exception as e:
            print(f"Error verifying image {img_path}: {e}")

    features = Features({
        "image": HFImage(),  
        "caption": Value("string"),
    })

    dataset = HFDataset.from_dict({"image": images, "caption": captions}, features=features)
        

    print(f"{max_samples} images + captions loaded...")
    

    image_column = "image"
    caption_column = "caption"
    
    #################################################



    def tokenize_captions_multi(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError("Invalid caption type")

        inputs = tokenizer(captions, padding="max_length", truncation=True, return_tensors="pt")
        inputs_1 = tokenizer([c[:len(c)//3] for c in captions], padding="max_length", truncation=True, return_tensors="pt")
        inputs_2 = tokenizer([c[len(c)//3:2*len(c)//3] for c in captions], padding="max_length", truncation=True, return_tensors="pt")
        inputs_3 = tokenizer([c[2*len(c)//3:] for c in captions], padding="max_length", truncation=True, return_tensors="pt")
        inputs_null = tokenizer([""] * len(captions), padding="max_length", truncation=True, return_tensors="pt")
        
        # print("len(captions): ", len(captions))



        return inputs.input_ids, inputs_1.input_ids, inputs_2.input_ids, inputs_3.input_ids, inputs_null.input_ids
    
    def preprocess_train_multi(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"], examples["input_ids_1"], examples["input_ids_2"], examples["input_ids_3"], examples["input_ids_null"] = tokenize_captions_multi(examples)
        return examples
    

    test_dataset = dataset.with_transform(preprocess_train_multi)

    
    from torch.utils.data import Subset


    subset_indices = range(min(len(test_dataset), 2500))
    subset_dataset = Subset(test_dataset, subset_indices)
    
    test_dataloader = torch.utils.data.DataLoader(
        subset_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=flags.train_batch_size,
        num_workers=flags.dataloader_num_workers,
    )

    return test_dataloader




@torch.no_grad()
def mi_mtcl_denoise(model, batch, vae, text_encoder, device, ):  # x_sec_list_s, x_sec_recon_list_s):
    global Noise
    global Noise_usedidx

    batch["pixel_values"] = batch["pixel_values"].to(device)

    latents = vae.encode(batch["pixel_values"].to(torch.float32)).latent_dist.sample()

    latents = latents * vae.config.scaling_factor


    T = flags.T
    even_num = flags.even_num
    max_n_samples = flags.max_n_samples

    max_clid_samples = flags.max_clid_samples

    start = T // 2 - (even_num * max_n_samples // 2)
    

    t_to_eval = list(range(start, T, even_num))[:max_n_samples]

    start_idx = len(t_to_eval) // 2 - max_clid_samples // 2

    ## -------------- Manual -------------- ##
    t_to_eval = np.array([140])
    t_clid_to_eval = t_to_eval.copy()
    ## ------------------------------------ ##

    noise = None

    batch_loss = {"cond0": [], "cond1_dif": [], "cond2_dif": [], 'cond3_dif': [], "condNull_dif": [], "Null": []}

    for latent, input_ids, input_ids_1, input_ids_2, input_ids_3, input_ids_null in zip(latents, batch["input_ids"],
                                                                                        batch['input_ids_1'],
                                                                                        batch['input_ids_2'],
                                                                                        batch['input_ids_3'],
                                                                                        batch['input_ids_null']):
        assert latent.shape[-3:] == (4, 64, 64)  ###te

        ts = torch.tensor(np.concatenate([t_to_eval] * flags.trials_eacht)).long()  ### flags.trials_eacht=1
        ts_other = torch.tensor(np.concatenate([t_clid_to_eval] * flags.trials_eacht)).long()  ### flags.trials_eacht=1

        pixel_mtcl = latent.view(-1, 4, 64, 64).expand(len(t_to_eval), 4, 64, 64)
        
        ## -------------- Random -------------- ##
        noise = Noise[Noise_usedidx: Noise_usedidx + len(t_to_eval)]
        noise_other = noise[[start_idx + i for i in list(range(max_clid_samples))]]
        


        x_mtcl = scheduler.add_noise(pixel_mtcl.to(device), noise.to(device), ts.to(device))

        input_id_mtcl = input_ids.expand(len(t_to_eval), -1)
        emd_mtcl = text_encoder(input_id_mtcl.to(device))[0]
        
        noise_pred_emd_ori = model(x_mtcl, ts.to(device), emd_mtcl).sample
        loss_emd_ori = F.mse_loss(noise_pred_emd_ori.float(), noise.float().to(device), reduction="mean")

        batch_loss["cond0"].append(float(loss_emd_ori.detach().cpu()))


        for i, (input_ids_other, dict_name) in enumerate(zip([input_ids_1, input_ids_2, input_ids_3, input_ids_null],
                                              ['cond1_dif', 'cond2_dif', 'cond3_dif', 'condNull_dif'])):
            pixel_mtcl_other = latent.view(-1, 4, 64, 64).expand(len(t_clid_to_eval), 4, 64, 64)
            
            x_mtcl_other = scheduler.add_noise(pixel_mtcl_other.to(device), noise_other.to(device), ts_other.to(device))
            input_id_mtcl_other = input_ids_other.expand(len(t_clid_to_eval), -1)
            emd_mtcl_other = text_encoder(input_id_mtcl_other.to(device))[0]

            noise_pred_emd_other = model(x_mtcl_other, ts_other.to(device), emd_mtcl_other).sample
            
            loss_emd_other = F.mse_loss(noise_pred_emd_other.float(), noise_other.float().to(device), reduction="mean")

            batch_loss[dict_name].append(float(loss_emd_other.detach().cpu()) - float(loss_emd_ori.detach().cpu()))
            
            if i+1 == 4:
                batch_loss["Null"].append(float(loss_emd_other.detach().cpu()))

        Noise_usedidx += len(t_to_eval)

    return batch_loss, t_to_eval


@torch.no_grad()
def mi_mtcl_denoise_emb(model, batch, vae, text_encoder, device, ):  # x_sec_list_s, x_sec_recon_list_s):
    global Noise
    global Noise_usedidx
    global anchor

    batch["pixel_values"] = batch["pixel_values"].to(device)

    latents = vae.encode(batch["pixel_values"].to(torch.float32)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    T = flags.T
    even_num = flags.even_num
    max_n_samples = flags.max_n_samples

    max_clid_samples = flags.max_clid_samples

    start = T // 2 - (even_num * max_n_samples // 2)
    t_to_eval = list(range(start, T, even_num))[:max_n_samples]

    start_idx = len(t_to_eval) // 2 - max_clid_samples // 2
   
    
    ## -------------- Manual -------------- ##
    if flags.anchor != 0:
        t_to_eval = np.array([flags.anchor*10])
    else:
        t_to_eval = np.array([140])
    
    t_clid_to_eval = t_to_eval.copy()
    ## ------------------------------------ ##


    noise = None


    batch_loss = {"cond0": [], "condNull_dif": [], "Null": []}

    for latent, embeds, input_ids_null in zip(latents, batch["text_embedding"], batch['input_ids_null']):
        assert latent.shape[-3:] == (4, 64, 64)  ###te

        ts = torch.tensor(np.concatenate([t_to_eval] * flags.trials_eacht)).long()  ### flags.trials_eacht=1
        # print(f"ts: {ts}")
        ts_other = torch.tensor(np.concatenate([t_clid_to_eval] * flags.trials_eacht)).long()  ### flags.trials_eacht=1

        pixel_mtcl = latent.view(-1, 4, 64, 64).expand(len(t_to_eval), 4, 64, 64)
        

        ## -------------- Certain Noise -------------- ##
        npy_path = "path/to/noise/used/during/optim/data/rnd_noise_1.npy"
        rnd_noise_np = np.load(npy_path)
        noise = torch.from_numpy(rnd_noise_np).to(device)
        noise_other = noise.detach().clone()
        


        x_mtcl = scheduler.add_noise(pixel_mtcl.to(device), noise.to(device), ts.to(device))
        
        

        emd_mtcl = embeds.expand(len(t_to_eval), -1, -1).clone() 
        emd_mtcl = emd_mtcl.to(device)
        
        noise_pred_emd_ori = model(x_mtcl, ts.to(device), emd_mtcl).sample
        loss_emd_ori = F.mse_loss(noise_pred_emd_ori.float(), noise.float().to(device), reduction="mean")

        batch_loss["cond0"].append(float(loss_emd_ori.detach().cpu()))
        

        for input_ids_other, dict_name in zip([input_ids_null],['condNull_dif']):
            pixel_mtcl_other = latent.view(-1, 4, 64, 64).expand(len(t_clid_to_eval), 4, 64, 64)
            x_mtcl_other = scheduler.add_noise(pixel_mtcl_other.to(device), noise_other.to(device), ts_other.to(device))
            input_id_mtcl_other = input_ids_other.expand(len(t_clid_to_eval), -1)
            emd_mtcl_other = text_encoder(input_id_mtcl_other.to(device))[0]


            noise_pred_emd_other = model(x_mtcl_other, ts_other.to(device), emd_mtcl_other).sample
            loss_emd_other = F.mse_loss(noise_pred_emd_other.float(), noise_other.float().to(device), reduction="mean")

            batch_loss[dict_name].append(float(loss_emd_other.detach().cpu()) - float(loss_emd_ori.detach().cpu()))
            
            ## NULL
            batch_loss["Null"].append(float(loss_emd_other.detach().cpu()))
            
            loss_emd_ = F.mse_loss(noise_pred_emd_other.float(), noise_pred_emd_ori.float().to(device), reduction="mean")


        Noise_usedidx += len(t_to_eval)

    return batch_loss, t_to_eval




torch.no_grad()

if Use_data_model_name.endswith("GT"):
    num = 3 ## GT (CLiD-based)
else:
    num = 1 ## MoFit (emb, VLM)

for Max_n_samples in [num, ]:  # [1, 3, 5, 7, 9]: default=3

    flags.max_n_samples = Max_n_samples
    flags.max_clid_samples = Max_n_samples

    T = flags.T
    even_num = flags.even_num
    max_n_samples = flags.max_n_samples
    start = T // 2 - (even_num * max_n_samples // 2)
    t_to_eval = list(range(start, T, even_num))[:max_n_samples]

    set_random_seed(seed=0)
    Noise = torch.randn(5000 * 40, 4, 64, 64)

    Noise_usedidx = 0
    # print('Noise.shape:', Noise.shape)

    loader_flag = 0
    output_paths = [] 
    for data_name in [flags.dataset_train_name, flags.dataset_test_name]: 

        print(f"Use_data_model_name: {Use_data_model_name} -> data_name: {data_name}")
        print()
        
        if loader_flag == 0:
            trainOrtest = "train"
            loader_flag += 1
        else:
            trainOrtest = "test"

        print("*** trainOrtest ***  ", trainOrtest)
        
        #############################################
        max_samples = 500 ## -1 for FULL
        print(f"Number of samples: {max_samples}")
        #############################################
        
        if Use_data_model_name == "LAION_emb" or Use_data_model_name == "Pokemon_emb" or Use_data_model_name == "flickr_8k_emb" or Use_data_model_name == "COCO_emb":
            loader = get_data_emb(flags, Use_data_model_name, data_name, max_samples=max_samples)
        else:
            loader = get_data(flags, Use_data_model_name, data_name, max_samples=max_samples) # max_samples=2500: default of the CLiD code
        assert flags.max_n_samples * len(loader) * 2 < Noise.shape[0]

        print("Calculating...")
        if Use_data_model_name == "LAION_emb" or Use_data_model_name == "Pokemon_emb" or Use_data_model_name == "flickr_8k_emb" or Use_data_model_name == "COCO_emb":
            dataset_loss_dict = {"cond0": [], "condNull_dif": [], "Null": []}
        else:
            dataset_loss_dict = {"cond0": [], "cond1_dif": [], "cond2_dif": [], 'cond3_dif': [], "condNull_dif": [], "Null": []}
            
        pbar = tqdm(loader)
        for step, batch in enumerate(pbar):

            model = unet
            if flags.attack == 'MoFit':
                if Use_data_model_name == "LAION_emb" or Use_data_model_name == "Pokemon_emb" or Use_data_model_name == "flickr_8k_emb" or Use_data_model_name == "COCO_emb":
                    batch_loss, t_to_eval = mi_mtcl_denoise_emb(model, batch, vae, text_encoder, device)
                else:
                    batch_loss, t_to_eval = mi_mtcl_denoise(model, batch, vae, text_encoder, device) 
                for key, value in batch_loss.items():
                    dataset_loss_dict[key].extend(value)

            else:
                print('Error, No implement!', flags.attack)
                exit()
                
            pbar.set_description(f"Considering t={t_to_eval.tolist()}...")

        path_temp = '/{}_{}images_{}_t_{}.txt'.format(Use_data_model_name,
                                                            (len(loader)-1)*flags.train_batch_size+batch["pixel_values"].shape[0],                                               
                                                            trainOrtest, 
                                                            t_to_eval.tolist())
    

        output_paths.append(path_temp)
        with open(flags.outdir + path_temp, 'w', encoding='utf8') as f:
            f.write(str(flags.__dict__) + '\t' + diff_path + '\t' + '\n')
            lines = ['\t'.join(map(lambda x: "{:.5g}".format(x), values)) for values in
                     zip(*dataset_loss_dict.values())]
            f.write('\n'.join(lines))

            

        print('save in', flags.outdir + path_temp)
        print()


    
