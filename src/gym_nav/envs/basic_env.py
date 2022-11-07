import base64
import io
import json
import math
from turtle import delay
import requests
import subprocess
import time
from typing import Dict, List

import gym
import numpy as np
from gym import spaces
from PIL import Image

N_DISCRETE_ACTIONS = 5
HEIGHT = 256
WIDTH = 256
N_CHANNELS = 3

request_types = {
    "ResetEnv": "json",
    "ChangeLevel": "json",
    "GetActionSpace": "json",
    "Reset": "json",
    "GetAgentAddress": "json",
    "PerformAction": "json",
    "GetVisualState": "bytes",
    "GetState": "json",
}

targets = ["Env", "Agent"]

errors = {
    404: "Not Found"
}


def read_img(b64_data):
    base64_bytes = b64_data.encode('ascii')
    image_bytes = base64.b64decode(base64_bytes)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(img)


class BasicEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, args):
        super(BasicEnv, self).__init__()
        # Store env args
        
        self.args = args

        render = "" if self.args["render"] else "-RenderOffscreen"
        self.process = subprocess.Popen(self.args["exe_loc"] + " -RCWebControlEnable -RCWebInterfaceEnable " + render)

        self.base_endpoint = f"http://{self.args['host']}:{self.args['port']}/remote/preset/MyPreset/function/"

        self.session = requests.Session()

        response = self.__make_request('EnvInfo')

        self.action_space = spaces.Discrete(response["ActionsSpace"])
        self.levels = response["Levels"]
        self.observation_space = spaces.Dict({
            "rgb_img": spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8),
            "angle": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "distance": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "sin": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "cos": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "target_reached": spaces.Box(low=False, high=True, shape=(1,), dtype=np.bool8),
            "has_collided": spaces.Box(low=False, high=True, shape=(1,), dtype=np.bool8),
        })
        
        spaces.Box(low=0, high=255, shape=(
            HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

    def __make_request(self, request: str, args: Dict = {}):
        # Check request_type
        #assert request in request_types, f"request type must be one of {request_types}"
        # Prepare request json
        pload = {
            "Parameters": args,
            "GenerateTransaction": True
        }
        # Send request
        r = self.session.put(self.base_endpoint + request,
                             data=json.dumps(pload))
        response = r.json()
        # Return response
        return response["ReturnedValues"][0]

    def __compute_reward(self, agent_state):
        reward = 0.0
        done = False
        if agent_state["has_collided"] and not agent_state["target_reached"]:
            reward = -3
            done = True
        if agent_state["target_reached"]:
            reward = 3
            done = True
        return reward, done

    def __get_state(self, data):

        img = read_img(data["RGBCapture"])

        state = {
            "rgb_img": img,
            "angle": np.array([data["Angle"]], dtype=np.float32),
            "distance": np.array([data["Distance"]], dtype=np.float32),
            "sin": np.array([math.sin(data["Angle"])], dtype=np.float32),
            "cos": np.array([math.cos(data["Angle"])], dtype=np.float32),
            "target_reached": np.array([data["TargetReached"]], dtype=np.bool8),
            "has_collided": np.array([data["HasCollided"]], dtype=np.bool8),
        }

        return state

    def step(self, action):
        # Execute one time step within the environment
        response = self.__make_request(
            "PerformAction", args={"AgentId": 0, "Action": action})

        # Process agent state
        state = self.__get_state(response)

        # Compute reward
        reward, done = self.__compute_reward(state)
        info = None

        return state, reward, done, info

    def reset(self, seed: int = None, return_info: bool = False, options: dict = None):
        response = self.__make_request("ResetEnv", args=options)

        time.sleep(0.1)

        state, _, _, _ = self.step(0)

        return state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        ...

    def close(self):
        self.process.kill()


if __name__ == "__main__":
    test_env = BasicEnv({"host": "localhost", "port": "8080",
                        "exe_loc": "D:/Midgard/Build/WindowsNoEditor/NavEnvironment.exe"})

    time.sleep(1)
    test_env.reset(
        {
            "Difficulty": 0.1,
            "LevelName": "FlatLevel",
            "FullReset": True
        }
    )

    time.sleep(1)
    test_env.step(1)

    time.sleep(1)
    test_env.step(0)

    time.sleep(1)
    test_env.reset()
