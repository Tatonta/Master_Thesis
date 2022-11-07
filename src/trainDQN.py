import time

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.models as models
from torchvision import transforms

from algorithms.DQN import DQN
#from algorithms.ppo import PPO
from utils.config import *
from utils.saver import TBSaver

# set device to cpu or cuda
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

# setup image transforms
preprocess = transforms.Compose([
    transforms.ToTensor() ,
    transforms.Normalize(mean = [0.485 , 0.456 , 0.406] ,
                         std = [0.229 , 0.224 , 0.225]) ,
])


class FeaturesExtractor(nn.Module):
    def __init__(self):
        super(FeaturesExtractor , self).__init__()
        model = torch.hub.load('pytorch/vision:v0.10.0' ,
                               'deeplabv3_mobilenet_v3_large' , pretrained = True)
        self.backbone = model.backbone

    def forward(self , x):
        out = self.backbone(x)['out']
        out = F.adaptive_avg_pool2d(out , 1)
        return out


def train():
    # setup logger
    logger = TBSaver(base_output_dir = exp_path , title = "PPO-Meadow" , args = {})

    # init the environment
    env = gym.make("gym_nav:basic-nav-v0" ,
                   args = {
                       "host": "localhost" ,
                       "port": "30010" ,
                       "exe_loc": "D:/MIDGARDv2/WindowsNoEditor/NavEnvironment.exe" ,
                       "render": True
                   })

    # state space dimension
    state_dim = 3  # env.observation_space.shape

    # action space dimension
    action_dim = env.action_space.n

    ################# training procedure ################

    # init features extractor

    model = FeaturesExtractor()
    model = model.to(device)
    model.eval()
    mode = "mixed"
    # initialize a PPO agent
    dqn_agent = DQN(state_dim , action_dim , lr_actor , lr_critic , gamma , K_epochs , EPS , EPS_END , EPS_DECAY ,
                    mode = mode)

    time_step = 0
    i_episode = 0

    difficulty = 0.1

    call_args = {"Difficulty": difficulty ,
                 "LevelName": "Meadow" , "FullReset": True}

    state = env.reset(options = call_args)

    # training loop
    while i_episode <= max_training_timesteps:
        if i_episode % 100 == 0 and i_episode > 0:
            difficulty += 0.05

        current_ep_reward = 0

        times = []

        state = env.reset(
            options = {"Difficulty": difficulty , "LevelName": "Meadow" , "SimulationSpeed": 2.0})

        states = [TF.to_tensor(state["rgb_img"]).unsqueeze(0)]

        time.sleep(2)
        print(f"Start episode {i_episode}.")

        for t in range(1 , 500):

            start = time.time()
            # store visual state for video generation
            states.append(TF.to_tensor(state["rgb_img"]).unsqueeze(0))

            img_state = preprocess(state["rgb_img"]).unsqueeze(0).to(device)
            with torch.no_grad():
                img_feat = model(img_state).view(1 , -1)

            compass_state = torch.tensor(
                (state["sin"] , state["cos"] , state["distance"])).view(-1).unsqueeze(0).to(device)

            action = dqn_agent.select_action(
                {'image': img_feat , 'compass': compass_state})
            next_state , reward , done , _ = env.step(action)
            next_state_img = preprocess(next_state["rgb_img"]).unsqueeze(0).to(device)
            with torch.no_grad():
                next_img_feat = model(next_state_img).view(1 , -1)
            next_state_cmp = torch.tensor(
                (next_state["sin"] , next_state["cos"] , next_state["distance"])).view(-1).unsqueeze(0).to(device)

            # next_state_pp = {"image":next_img_feat,"compass": next_state_cmp}

            # saving reward and is_terminals
            dqn_agent.store_transition(img_feat, compass_state, action, reward, next_img_feat, next_state_cmp, done)

            time_step += 1
            current_ep_reward += reward
            state = next_state
            # update PPO agent
            if time_step % update_timestep == 0:
                dqn_agent.update(i_episode)

            if done:
                break

            end = time.time()
            times.append(end - start)

        logger.dump_metric(current_ep_reward , i_episode ,
                           "train" , "episode_reward")

        logger.dump_metric(t , i_episode ,
                           "train" , "episode_steps")

        if i_episode % log_freq == 0:
            t_states = torch.cat(states).unsqueeze(0)
            logger.dump_batch_video(
                t_states , i_episode , "train" , "episode_anim")

        i_episode += 1

    env.close()


if __name__ == '__main__':
    train()
