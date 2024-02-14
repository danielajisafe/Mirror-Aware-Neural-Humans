

import os
import pdb
import sys
import json
from glob import glob

sys.path.append(".")
from core_mirror.transforms import alpha_detections_to_dcpose_form_non_eval, drop_bystanders_non_eval, alpha_detections_to_dcpose_form_mirr_eval, drop_bystanders_mirr_eval


def combine_annots(annot_path, cam_id=3):
    """annot_path: path to /annots folder
    camera_ids = (2,3,4,5,6,7)
    """

    files_path = f"{annot_path}/{cam_id}/*.json"
    json_files = sorted(glob(files_path))

    camera_comb_data = {"Info" : []}
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
            frame_data = data["annots"]
            frame_img_url = data["filename"].replace("images", "dataset/zju-m-seq1/images")
            # update frame url to 0-indexed
            frame_img_id_str = frame_img_url.split("/")[-1].split(".")[0]
            json_file_id = int(json_file.split("/")[-1].split(".")[0])
            frame_img_url = frame_img_url.replace(frame_img_id_str, f"{json_file_id:08d}")

            for fdata in frame_data:
                fdict = {"image_path": frame_img_url,
                            "bbox": fdata["bbox"],
                            "keypoints": fdata["keypoints"]
                            }
                camera_comb_data["Info"].append(fdict)
    
    save_filepath = f"{annot_path}/combined/{cam_id}"
    os.makedirs(f"{save_filepath}", exist_ok=True)
    with open(f"{save_filepath}/{cam_id}.json", 'w') as f:
        json.dump(camera_comb_data, f)
    print(f"combined data saved to {save_filepath}/{cam_id}.json\n")


if __name__ == "__main__":
    cam_id=3

    # Combining the annotation json files for mirror-eval dataset
    annot_path = "dataset/zju-m-seq1/annots"
    combine_annots(annot_path, cam_id=cam_id)

    # Converting alphapose detections (eval) to dcpose format
    filepath = "dataset/eval"
    alpha_detections_to_dcpose_form_mirr_eval(cam_id=cam_id, filepath=filepath)
    drop_bystanders_mirr_eval(cam_id=cam_id, filepath=filepath)

    file_dir = "dataset/non_eval"
    # Converting alphapose detections (non-eval) to dcpose format
    alpha_detections_to_dcpose_form_non_eval(file_dir=file_dir)
    drop_bystanders_non_eval(file_dir=file_dir)