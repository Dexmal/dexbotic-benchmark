import requests
import cv2
import numpy as np


class RoboTwin2VLAAgent:
    def __init__(
        self, 
        server_url="http://localhost:7891", 
        use_cameras=["head_camera_rgb"],
        action_horizon=-1, 
        action_mode="relative",
    ):
        self.server_url = f"{server_url}/process_frame"
        print(f"Server url: {self.server_url}")
        self.use_cameras = use_cameras
        print(f"Use cameras: {self.use_cameras}")
        self.action_horizon = action_horizon
        print(f"Action horizon: {self.action_horizon}")
        assert action_mode in ["relative", "delta"], "Only support 'relative' or 'delta' action mode."
        self.action_mode = action_mode
        print(f"Action mode: {self.action_mode}")

    def get_action(self, instruction: str, rgbs: np.ndarray) -> np.ndarray: 
        encoded_images = [cv2.imencode('.png', rgb)[1].tobytes() for rgb in rgbs]
        ret = requests.post(
            self.server_url,
            data={"text": instruction, "temperature": 1.0},
            files=[("image", _img) for _img in encoded_images],
        )
        raw_action = ret.json().get('response')
        action_chunk = np.array(raw_action)

        if self.action_horizon > 0 and len(action_chunk) > self.action_horizon:
            action_chunk = action_chunk[:self.action_horizon]
        return action_chunk

    def convert_delta_to_absolute_action(self, state, action_chunk: np.ndarray) -> np.ndarray:
        left_arm_state = state[:6]
        right_arm_state = state[7:13]

        left_arm_action = action_chunk[:, :6]
        left_gripper_action = action_chunk[:, 6:7]
        right_arm_action = action_chunk[:, 7:13]
        right_gripper_action = action_chunk[:, 13:14]

        left_arm_cumsum = np.cumsum(left_arm_action, axis=0)
        right_arm_cumsum = np.cumsum(right_arm_action, axis=0)

        left_arm_absolute = left_arm_state + left_arm_cumsum
        right_arm_absolute = right_arm_state + right_arm_cumsum

        absolute_action = np.concatenate(
            [left_arm_absolute, left_gripper_action, right_arm_absolute, right_gripper_action], axis=1
        )
        return absolute_action

    def convert_relative_to_absolute_action(self, state, action_chunk: np.ndarray) -> np.ndarray:
        left_arm_state = state[:6]
        right_arm_state = state[7:13]

        left_arm_action = action_chunk[:, :6]
        left_gripper_action = action_chunk[:, 6:7]
        right_arm_action = action_chunk[:, 7:13]
        right_gripper_action = action_chunk[:, 13:14]

        left_arm_absolute = left_arm_state + left_arm_action
        right_arm_absolute = right_arm_state + right_arm_action

        absolute_action = np.concatenate(
            [left_arm_absolute, left_gripper_action, right_arm_absolute, right_gripper_action], axis=1
        )
        return absolute_action


def unittest_request_cogact(): 
    image = np.ones((480, 640, 3), dtype=np.uint8)
    prompt = "Do something."
    url = "http://localhost:7891/process_frame"
    encoded_images = [cv2.imencode('.png', image)[1].tobytes() for image in [image, ]]
    ret = requests.post(
        url,
        data={"text": prompt, "temperature": 1.0},
        files=[("image", _img) for _img in encoded_images],
    )
    raw_action = ret.json().get('response')

    action_chunk = np.array(raw_action)
    print(f"raw_action: {action_chunk}")


if __name__ == "__main__":
    unittest_request_cogact()