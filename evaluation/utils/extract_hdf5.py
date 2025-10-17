from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import h5py
import json
import numpy as np
import cv2
import io
import imageio.v2 as imageio
import megfile
import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_dir_path", type=str, default="")
    parser.add_argument("--output_dir_path", type=str, default="")
    return parser.parse_args()

max_threads = min(int(os.environ.get('NUM_PROCESS', 32)), (os.cpu_count() or 8) + 4)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        return super().default(obj)
    
    
class ProcessHDF5:
    """ Process a directory of hdf5 files and output to a new directory,
        which contain a sub-directory of videos, and a sub-directory of other metadata
        """


    def __init__(self, hdf5_dir_path, output_dir_path):
        self.hdf5_dir_path = hdf5_dir_path
        self.output_dir_path = output_dir_path

    def extract_task_name_and_language(self, hdf5_file_path):
        # extract the task name from the hdf5 file name
        # file name is like: STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo
        # the expected task name is: pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy
        
        task_name = hdf5_file_path.split("/")[-1].replace("_demo.hdf5", "")
        task_name = re.sub(r"^[A-Z0-9_]+", "", task_name)
        
        language = task_name.replace("_", " ")
        
        return task_name, language
        
    def process_func(self, hdf5_file_path):
        # structure of the hdf5 file:
        # data/
        #   - demo_0/
        #     - actions
        #     - robot_states
        #     - states
        #     - obs
        #       - agentview_rgb: [video_length 128 128 3]
        #       - eye_in_hand_rgb: [video_length 128 128 3]
        #       - ee_states : [video_length 6]
        #       - gripper_states : [video_length 2] I only use the first value, since the second value has the same absolute value but opposite sign
        #       - joint_states  : [video_length 7]
        #   - demo_1/-
        #   - ...
        
        # Structure of the output directory:
        # output/
        #   - json/
        #       - task_name_0/ the task name could be extracted from the hdf5 file name
        #         - episode_0.json
        #         - episode_1.json
        #         - ...
        #   - video/
        #       - task_name_0/
        #         - episode_0.mp4
        #         - episode_1.mp4
        
        task_name, language = self.extract_task_name_and_language(hdf5_file_path)
        
        # create the output directory
        output_json_dir = os.path.join(self.output_dir_path, "json", task_name)
        output_video_dir = os.path.join(self.output_dir_path, "video", task_name)
        
        num_noops = 0
        # read the hdf5 file
        with h5py.File(hdf5_file_path, "r") as f:
            for episode_id in f['data'].keys():
                demo_json_path = os.path.join(output_json_dir, f"{episode_id}.json")
                demo_front_video_path = os.path.join(output_video_dir, f"{episode_id}_front.mp4")
                demo_eye_video_path = os.path.join(output_video_dir, f"{episode_id}_eye.mp4")
                
                data_dicts = {
                    "agentview_rgb": self.resize_to_256(self.rotate_image_180(f['data'][episode_id]['obs']['agentview_rgb'][()])),
                    "eye_in_hand_rgb": self.resize_to_256(self.rotate_image_180(f['data'][episode_id]['obs']['eye_in_hand_rgb'][()])),
                    "ee_states": f['data'][episode_id]['obs']['ee_states'][()],
                    "gripper_states": f['data'][episode_id]['obs']['gripper_states'][()],
                    "joint_states": f['data'][episode_id]['obs']['joint_states'][()],
                    "actions": f['data'][episode_id]['actions'][()],
                    "robot_states": f['data'][episode_id]['robot_states'][()],
                    "states": f['data'][episode_id]['states'][()]
                }
                data_dicts, _num_noops = self.filter_noops(data_dicts)
                num_noops += _num_noops
                
                json_data = {
                    "ee_states": data_dicts['ee_states'],
                    "gripper_states": data_dicts['gripper_states'],
                    "joint_states": data_dicts['joint_states'],
                    "actions": data_dicts['actions'],
                    "robot_states": data_dicts['robot_states'],
                    "states": data_dicts['states'],
                    "front_video_path": demo_front_video_path,
                    "eye_video_path": demo_eye_video_path,
                    "language": language,
                    "task_name": task_name,
                    "episode_id": episode_id,
                }
                
                # image to video
                self.image_to_video(data_dicts['agentview_rgb'], demo_front_video_path)
                self.image_to_video(data_dicts['eye_in_hand_rgb'], demo_eye_video_path)
                
                # save the json file
                with megfile.smart_open(demo_json_path, "w") as f1:
                    json.dump(json_data, f1, indent=2, cls=NumpyEncoder)
        print(f"Number of noops in {hdf5_file_path}: {num_noops}")
    
    def filter_noops(self, data_dicts):
        def is_noop(action, prev_action=None, threshold=1e-4):
            """
            Returns whether an action is a no-op action.

            A no-op action satisfies two criteria:
                (1) All action dimensions, except for the last one (gripper action), are near zero.
                (2) The gripper action is equal to the previous timestep's gripper action.

            Explanation of (2):
                Naively filtering out actions with just criterion (1) is not good because you will
                remove actions where the robot is staying still but opening/closing its gripper.
                So you also need to consider the current state (by checking the previous timestep's
                gripper action as a proxy) to determine whether the action really is a no-op.
            """
            # Special case: Previous action is None if this is the first action in the episode
            # Then we only care about criterion (1)
            if prev_action is None:
                return np.linalg.norm(action[:-1]) < threshold

            # Normal case: Check both criteria (1) and (2)
            gripper_action = action[-1]
            prev_gripper_action = prev_action[-1]
            return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action
        
        # Filter out no-op actions
        filtered_data_dicts = defaultdict(list)
        prev_action = None
        num_noops = 0
        for time_step in range(data_dicts['actions'].shape[0]):
            if is_noop(data_dicts['actions'][time_step], prev_action):
                num_noops += 1
                continue
            prev_action = data_dicts['actions'][time_step]
            for key, value in data_dicts.items():
                filtered_data_dicts[key].append(value[time_step])
        filtered_data_dicts = {k: np.stack(v) for k, v in filtered_data_dicts.items()}
        return filtered_data_dicts, num_noops
            
    def rotate_image_180(self, image_array):
        # rotate the image array 180 degrees
        # image_array is a 4D array, [video_length, height, width, 3]
        # return the rotated image array
        return np.flip(image_array, axis=(1, 2))
    
    def resize_to_256(self, image_array):
        # resize the image array to 256x256
        # image_array is a 4D array, [video_length, height, width, 3]
        # return the resized image array
        resized = []
        for i in range(image_array.shape[0]):
            resized_img = cv2.resize(image_array[i], (256, 256), interpolation=cv2.INTER_CUBIC)
            resized.append(resized_img)
            
        return np.stack(resized, axis=0)
    
    def image_to_video(self, image_array, output_video_path):
        # convert the image array to a video
        # image_array is a 4D array, [video_length, height, width, 3]
        # output_video_path is the path to the output video
        
        # create the video writer
        buffer = io.BytesIO()
        imageio.mimwrite(buffer, image_array, format='mp4', fps=30, codec='h264')
        
        buffer.seek(0)
        with megfile.smart_open(output_video_path, "wb") as f:
            f.write(buffer.read())
            
    def process_data(self):
        # scan all hdf5 files in the directory
        hdf5_files = [f for f in os.listdir(self.hdf5_dir_path) if f.endswith(".hdf5")]
        
        # debug using the first hdf5 files
        # self.process_func(os.path.join(self.hdf5_dir_path, hdf5_files[0]))
        
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = [executor.submit(self.process_func, os.path.join(self.hdf5_dir_path, f)) for f in hdf5_files]
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing hdf5 files"):
                future.result()

if __name__ == "__main__":
    args = parse_args()
    hdf5_dir_path = args.hdf5_dir_path
    output_dir_path = args.output_dir_path
    process_hdf5 = ProcessHDF5(hdf5_dir_path, output_dir_path)
    process_hdf5.process_data()