import os
from pathlib import Path
import numpy as np
import argparse
import tqdm
import pickle
from functools import reduce
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import transform_matrix

# Keep the necessary helper functions: get_available_scenes, get_sample_data, quaternion_yaw, obtain_sensor2top

def fill_test_infos(data_path, nusc, test_scenes, max_sweeps=10, with_camera=False):
    test_nusc_infos = []
    progress_bar = tqdm.tqdm(total=len(nusc.sample), desc="create_info", dynamic_ncols=True)

    ref_chan = "LIDAR_TOP"
    chan = "LIDAR_TOP"

    for index, sample in enumerate(nusc.sample):
        progress_bar.update()

        ref_sd_token = sample["data"][ref_chan]
        ref_sd_rec = nusc.get("sample_data", ref_sd_token)
        ref_cs_rec = nusc.get("calibrated_sensor", ref_sd_rec["calibrated_sensor_token"])
        ref_pose_rec = nusc.get("ego_pose", ref_sd_rec["ego_pose_token"])
        ref_time = 1e-6 * ref_sd_rec["timestamp"]

        ref_lidar_path, ref_boxes, _ = get_sample_data(nusc, ref_sd_token)

        ref_cam_front_token = sample["data"]["CAM_FRONT"]
        ref_cam_path, _, ref_cam_intrinsic = nusc.get_sample_data(ref_cam_front_token)

        ref_from_car = transform_matrix(ref_cs_rec["translation"], Quaternion(ref_cs_rec["rotation"]), inverse=True)
        car_from_global = transform_matrix(ref_pose_rec["translation"], Quaternion(ref_pose_rec["rotation"]), inverse=True)

        info = {
            "lidar_path": Path(ref_lidar_path).relative_to(data_path).__str__(),
            "cam_front_path": Path(ref_cam_path).relative_to(data_path).__str__(),
            "cam_intrinsic": ref_cam_intrinsic,
            "token": sample["token"],
            "sweeps": [],
            "ref_from_car": ref_from_car,
            "car_from_global": car_from_global,
            "timestamp": ref_time,
        }

        # Add camera information if required
        if with_camera:
            # Add camera info processing here (similar to the original script)
            pass

        # Process sweeps
        sample_data_token = sample["data"][chan]
        curr_sd_rec = nusc.get("sample_data", sample_data_token)
        sweeps = []
        while len(sweeps) < max_sweeps - 1:
            if curr_sd_rec["prev"] == "":
                if len(sweeps) == 0:
                    sweep = {
                        "lidar_path": Path(ref_lidar_path).relative_to(data_path).__str__(),
                        "sample_data_token": curr_sd_rec["token"],
                        "transform_matrix": None,
                        "time_lag": curr_sd_rec["timestamp"] * 0,
                    }
                    sweeps.append(sweep)
                else:
                    sweeps.append(sweeps[-1])
            else:
                curr_sd_rec = nusc.get("sample_data", curr_sd_rec["prev"])
                # Process sweep (similar to the original script)
                # ...

        info["sweeps"] = sweeps

        if sample["scene_token"] in test_scenes:
            test_nusc_infos.append(info)

    progress_bar.close()
    return test_nusc_infos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, help="Path to the nuScenes dataset.")
    parser.add_argument("--output_root", required=True, help="Output path where processed information located.")
    parser.add_argument("--max_sweeps", default=10, type=int, help="Max number of sweeps. Default: 10.")
    parser.add_argument("--with_camera", action="store_true", default=False, help="Whether use camera or not.")
    config = parser.parse_args()

    print(f"Loading nuScenes tables for version v1.0-test...")
    nusc_test = NuScenes(version="v1.0-test", dataroot=config.dataset_root, verbose=False)
    available_scenes_test = get_available_scenes(nusc_test)
    available_scene_names_test = [s["name"] for s in available_scenes_test]
    print("total scene num:", len(nusc_test.scene))
    print("exist scene num:", len(available_scenes_test))

    test_scenes = splits.test
    test_scenes = set([available_scenes_test[available_scene_names_test.index(s)]["token"] for s in test_scenes])

    print(f"Filling test information...")
    test_nusc_infos = fill_test_infos(config.dataset_root, nusc_test, test_scenes, max_sweeps=config.max_sweeps, with_camera=config.with_camera)

    print(f"Saving nuScenes test information...")
    os.makedirs(os.path.join(config.output_root, "info"), exist_ok=True)
    print(f"test sample: {len(test_nusc_infos)}")
    with open(os.path.join(config.output_root, "info", f"nuscenes_infos_{config.max_sweeps}sweeps_test.pkl"), "wb") as f:
        pickle.dump(test_nusc_infos, f)