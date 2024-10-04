import json
import numpy as np
from PIL import Image, ImageDraw


def load_and_draw(
        image_path,
        points_list,
        sizes_list=None,
        return_type="np",
):
    img = Image.open(image_path).convert('RGBA')
    return draw_on_image(img, points_list, sizes_list, return_type=return_type)


def draw_on_image(
        img,
        points_list,
        sizes_list=None,
        return_type="np",
):
    img = img.convert('RGBA')
    txt = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt)
    for i in range(len(points_list) // 2):
        handle, target = points_list[2 * i], points_list[2 * i + 1]
        radius = 2
        draw.ellipse(
            [
                handle[0] - radius, handle[1] - radius,
                handle[0] + radius, handle[1] + radius
            ],
            outline='red', width=2
        )
        draw.ellipse(
            [
                target[0] - radius, target[1] - radius,
                target[0] + radius, target[1] + radius
            ],
            outline='blue', width=2
        )
        draw.line([handle[0], handle[1], target[0], target[1]], fill='white', width=1)
        if sizes_list is not None:
            draw.rectangle(
                xy = [
                    handle[0] - sizes_list[i][0], handle[1] - sizes_list[i][1],
                    handle[0] + sizes_list[i][0], handle[1] + sizes_list[i][1],
                ],
                outline='red',
            )

    out = Image.alpha_composite(img, txt)
    if return_type == "np":
        return np.array(img.convert("RGB")), np.array(out.convert("RGB"))
    elif return_type == "pil":
        return img.convert("RGB"), out.convert("RGB")
    else:
        raise NotImplementedError


def get_concat_h(im1, im2):
    """
    concatenate horizontally 2 PIL images
    """
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def overlap_mask(
        img: Image,
        mask: np.array,  # binary mask, 1 is inside
):
    # Ensure mask is in the correct format for alpha compositing
    if (mask.ndim == 2) and (mask.max() < 2):  # Grayscale mask
        mask_img = Image.fromarray(mask * 100 + 100).convert("L")  # Convert to a single channel 'L' image
        mask_img = mask_img.convert("RGBA")
    else:
        raise ValueError("Unsupported mask format. Expected 2D and [0,1] mask")

    # Resize mask to image size if necessary
    if mask_img.size != img.size:
        raise NotImplementedError(f"Mask and Image has different size! {mask_img.size} and {img.size}")
    
    mask_img.putalpha(100)
    # Make sure img is in 'RGBA' mode for alpha compositing
    img = img.convert("RGBA")

    # Perform alpha compositing
    result = Image.alpha_composite(img, mask_img)

    return result


def load_points_from_theater_json(
        path_to_json,
        known_scene_id=9,
        novel_scene_id=7,
        canvas_hw=(512, 512),
):
    with open(path_to_json, "r") as f:
        data = json.load(f)
    default_wh = [x['wh'] for x in data['billboards']]
    half_wh_pixels = [
        [
            int(np.rint(x[0] * canvas_hw[1] / 4)),  # w=2 -> side=512, we need half side 256 => w * 512 / 4
            int(np.rint(x[1] * canvas_hw[0] / 4)),
        ]
        for x in default_wh
    ]

    known_positions = [
        [
            # transform (x, y) to regular image coordinate frame (top left corner) with pixels
            int(np.rint((x[0] + 1) * canvas_hw[1] / 2)),
            int(np.rint((1 - x[1]) * canvas_hw[0] / 2)),
        ]
        for x in data['scenes'][known_scene_id]['positions']
    ]
    print(known_positions)
    print(" | | |")

    new_positions = [
        [
            # transform (x, y) to regular image coordinate frame (top left corner) with pixels
            int(np.rint((x[0] + 1) * canvas_hw[1] / 2)),
            int(np.rint((1 - x[1]) * canvas_hw[0] / 2)),
        ]
        for x in data['scenes'][novel_scene_id]['positions']
    ]
    print(new_positions)

    points_list = []
    sizes_list = []
    for i in range(len(known_positions)):
        if half_wh_pixels[i][0] < canvas_hw[0]/2:
            points_list.append(
                known_positions[i]
            )
            points_list.append(
                new_positions[i]
            )
            sizes_list.append(
                half_wh_pixels[i]
            )
    return points_list, sizes_list


if __name__ == "__main__":
    points_l, sizes_l = load_points_from_theater_json(
        "/home/ivan/projects/dreamslicer/data/portrait42/paper_theater_data.json",
    )
    orig, drawn = load_and_draw(
        "/home/ivan/projects/dreamslicer/data/portrait42/image.png",
        points_list=points_l,
        sizes_list=sizes_l,
    )
    print(orig)
