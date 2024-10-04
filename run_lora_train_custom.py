"""
Usage from CLI:
python run_lora_train_custom.py -i path/to/image.png -p "text prompt" -o path/to/output/
python run_lora_train_custom.py -i lion512.png -o lora_lion/
python run_lora_train_custom.py -i dragbench_head.png -o lora_head/
python run_lora_train_custom.py -i portrait42.png -o lora_portrait/

"""    

import pickle
import PIL
from utils.lora_utils import train_lora
import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse PNG image and text prompt.")
    parser.add_argument(
        "--image", "-i",
        type=str, 
        help="The path to the input PNG image.",
        default="lion_stock.jpg"
        )
    parser.add_argument(
        "--prompt", "-p",
        type=str, 
        help="The text prompt for training LoRA.",
        default="",
        )
    parser.add_argument(
        "--out", "-o",
        type=str, 
        help="The folder to save the output.",
        default="lora_tmp/"
        )

    args = parser.parse_args()

    image_path = args.image
    text_prompt = args.prompt
    out_path = args.out
    print(image_path, text_prompt, out_path)

    # Open the image using PIL and convert it to a NumPy array
    image = PIL.Image.open(image_path).convert("RGB")
    image = np.array(image)

    train_lora(
        image=image,
        prompt=text_prompt,
        model_path="runwayml/stable-diffusion-v1-5",
        save_lora_path=out_path,
        lora_step=80,
    )
    
    