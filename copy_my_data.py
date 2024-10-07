import os
from shutil import copy2


if __name__ == "__main__":
    main_folder = "/home/ivan/projects/dreamslicer/data/"
    out_folder = "my_data/"
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
        copy2(
            os.path.join(datadir, "image.png"),
            outdir,
        )

        copy2(
            os.path.join(datadir, "paper_theater_data.json"),
            outdir,
        )