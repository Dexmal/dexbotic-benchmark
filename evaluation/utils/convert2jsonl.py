import os
import json
import megfile
import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir_path", type=str, default="")
    parser.add_argument("--output_dir_path", type=str, default="")
    return parser.parse_args()

max_threads = min(int(os.environ.get('NUM_PROCESS', 32)), (os.cpu_count() or 8) + 4)

class ProcessLibero:
    def __init__(self, input_dir_path, output_dir_path):
        self.input_dir_path = input_dir_path
        self.output_dir_path = output_dir_path
        
    def process_data(self):
        # scan all json files in the directory
        json_files = megfile.smart_glob(os.path.join(self.input_dir_path, "**", "*.json"))
        
        # debug using the first hdf5 files
        # self.process_func(json_files[0])
        
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = [executor.submit(self.process_func, f) for f in json_files]
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing json files"):
                future.result()
        
    
    def process_func(self, json_file_path):
        output_jsonl_path = json_file_path.replace(".json", ".jsonl").replace(self.input_dir_path, self.output_dir_path)
        # read the json file
        with megfile.smart_open(json_file_path, "r") as f:
            data = json.load(f)
            
        # convert the action to state
        initial_state = np.zeros_like(data["actions"][0])
        initial_state[-1] = 1.0
        states = [initial_state]
        for action in data["actions"][:-1]:
            state = action + states[-1]
            state[-1] = action[-1]
            states.append(state)
        states = np.stack(states, axis=0)
        
        episode_length = states.shape[0]
        episode_list = []
        for i in range(episode_length):
            episode_data = {
                "state": states[i].tolist(),
                "images_1": {'type': 'video', 'url': data["front_video_path"], 'frame_idx': i},
                "images_2": {'type': 'video', 'url': data["eye_video_path"], 'frame_idx': i},
                "prompt": data["language"],
                "is_robot": True
            }
            episode_list.append(episode_data)

        with megfile.smart_open(output_jsonl_path, "w", encoding="utf-8") as f:
            for episode_data in episode_list:
                f.write(json.dumps(episode_data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    args = parse_args()
    input_dir_path = args.input_dir_path
    output_dir_path = args.output_dir_path
    process_libero = ProcessLibero(input_dir_path, output_dir_path)
    process_libero.process_data()