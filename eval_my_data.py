import os
from run_drag_custom import my_run


if __name__ == "__main__":
    main_folder = "/home/ivan/projects/dreamslicer/data/"
    out_folder = "results/portraits_lora_rectangles/"
    for foldername in os.listdir(main_folder):
        if not os.path.isdir(os.path.join(main_folder, foldername)):
            continue
        if "paper_theater_data.json" not in os.listdir(os.path.join(main_folder, foldername)):
            continue
        print("\n\n--------")
        print(foldername)

        if "image.png" not in os.listdir(os.path.join(main_folder, foldername)):
            print("NO IMAGE")
            continue

        datadir = os.path.join(main_folder, foldername)
        outdir = os.path.join(out_folder, foldername)
        os.makedirs(outdir, exist_ok=True)

        for i in range(7, 14):
            my_run(
                input_image_path=os.path.join(datadir, "image.png"),
                json_data_path=os.path.join(datadir, "paper_theater_data.json"),
                lora_path=os.path.join(datadir, "LoRA/"),
                known_scene_id=9,
                new_scene_id=i,
                mask=None,
                use_rectangles=True,
                save_result_as=os.path.join(outdir, f"out{i}.png"),
                save_log_as=os.path.join(outdir, f"log_{i}.png"),
                save_result_drawing_as=os.path.join(outdir, f"b{i}.png"),
            )
        #     break
        # break
