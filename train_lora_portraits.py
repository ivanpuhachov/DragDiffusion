import os
import PIL
import numpy as np
from utils.lora_utils import train_lora
from utils.train_lora_v070 import train_lora as train_lora_v070


if __name__ == "__main__":
    main_folder = "/home/ivan/projects/dreamslicer/data/"
    for foldername in os.listdir(main_folder):
        if not os.path.isdir(os.path.join(main_folder, foldername)):
            continue
        if "paper_theater_data.json" not in os.listdir(os.path.join(main_folder, foldername)):
            continue
        print(foldername)

        if "image.png" not in os.listdir(os.path.join(main_folder, foldername)):
            print("NO IMAGE")
            continue

        image = PIL.Image.open(os.path.join(main_folder, foldername, "image.png")).convert("RGB")
        image = np.array(image)

        out_lora_folder = os.path.join(main_folder, foldername, "LoRA")
        # out_lora_folder = os.path.join(main_folder, foldername, "LoRA_0170")

        train_lora(
            image=image,
            prompt="",
            model_path="runwayml/stable-diffusion-v1-5",
            save_lora_path=out_lora_folder,
            lora_step=80,
        )
