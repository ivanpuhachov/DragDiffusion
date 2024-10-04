import json
import numpy as np
import matplotlib.pyplot as plt


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

    new_positions = [
        [
            # transform (x, y) to regular image coordinate frame (top left corner) with pixels
            np.rint((x[0] + 1) * canvas_hw[1] / 2),
            np.rint((1 - x[1]) * canvas_hw[0] / 2),
        ]
        for x in data['scenes'][novel_scene_id]['positions']
    ]

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
