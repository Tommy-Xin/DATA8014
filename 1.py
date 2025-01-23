import demo_util
import numpy as np
import torch
from PIL import Image
import imagenet_classes
from IPython.display import display
import os
from huggingface_hub import hf_hub_download
from modeling.maskgit import ImageBert
from modeling.titok import TiTok
import time

# supported tokenizer: [tokenizer_titok_l32_imagenet, tokenizer_titok_b64_imagenet, tokenizer_titok_s128_imagenet]
#titok_tokenizer = TiTok.from_pretrained("yucornetto/tokenizer_titok_l32_imagenet", cache_dir="/root/users/jusjus/1d-tokenizer/jusjus")
#titok_tokenizer = TiTok.from_pretrained("yucornetto/tokenizer_titok_l32_imagenet", cache_dir="/root/users/jusjus/1d-tokenizer/jusjus")
import demo_util
config = demo_util.get_config("/root/users/jusjus/1d-tokenizer/configs/infer/titok_bl128_vae_c16.yaml")
titok_tokenizer = TiTok.from_pretrained("yucornetto/tokenizer_titok_bl128_vae_c16_imagenet", cache_dir="/root/users/jusjus/1d-tokenizer/jusjus", config=config)

titok_tokenizer.eval()
titok_tokenizer.requires_grad_(False)
device = "cuda"

titok_tokenizer = titok_tokenizer.to(device)

# Tokenize an Image into 32 discrete tokens

def tokenize_and_reconstruct(img_path):
    original_image = Image.open(img_path)
    original_image = original_image.resize((256, 256))
    image = torch.from_numpy(np.array(original_image).astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
    if titok_tokenizer.quantize_mode == "vq":
        encoded_tokens = titok_tokenizer.encode(image.to(device))[1]["min_encoding_indices"]
    elif titok_tokenizer.quantize_mode == "vae":
        posteriors = titok_tokenizer.encode(image.to(device))[1]
        encoded_tokens = posteriors.sample()
    else:
        raise NotImplementedError
    print(encoded_tokens.shape)
    reconstructed_image = titok_tokenizer.decode_tokens(encoded_tokens)
    reconstructed_image = torch.clamp(reconstructed_image, 0.0, 1.0)
    reconstructed_image = (reconstructed_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
    reconstructed_image = Image.fromarray(reconstructed_image)

    return original_image,reconstructed_image

def save_image(image, img_idx, path):
    image.save(f"{path}/{img_idx}")

data_dir = "/root/users/jusjus/1d-tokenizer/jusjus/FLAME"
target_dir = "/root/users/jusjus/1d-tokenizer/jusjus/saves/FLAME_reconstructed"
os.makedirs(target_dir, exist_ok=True)

start_time_all = time.time()
for img in os.listdir(data_dir):
    if not img.endswith(".jpg") or img.endswith(".png"):
        continue
    #img_name = "FLAME.jpg"
    start_time = time.time()
    img_name = img
    img_path0 = f"{data_dir}/{img_name}"
    img_path = "/root/users/jusjus/1d-tokenizer/assets/ILSVRC2012_val_00010240.png"
    original_image0 = Image.open(img_path0)
    print(f"Shape of img_path0: {original_image0.size}")
    original_image1 = Image.open(img_path)
    print(f"Shape of img_path: {original_image1.size}")

    original_image, reconstructed_image = tokenize_and_reconstruct(img_path0)
    concatenated_image = Image.new('RGB', (original_image.width + reconstructed_image.width, original_image.height))
    concatenated_image.paste(original_image, (0, 0))
    concatenated_image.paste(reconstructed_image, (original_image.width, 0))
    save_image(concatenated_image, img_name, target_dir)
    end_time = time.time()
    print(f"Time taken to tokenize and reconstruct the image {img_name}: {end_time - start_time} seconds")
    break

end_time_all = time.time()
print(f"Total time taken to tokenize and reconstruct all images: {end_time_all - start_time_all} seconds")