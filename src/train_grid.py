from datetime import datetime

import torch

from environment import GridEnvironment
from ppo import PPO
from utils.config import *
from utils.saver import TBSaver

#import roboschool
#import pybullet_envs

MODE = "mixed"

################################### Training ###################################


def train():
    logger = TBSaver(base_output_dir=exp_path, title=env_name, args={})

    # init the environment
    env = GridEnvironment(
        steps_threshold=0, visual_debug=False, state_mode=MODE)

    # state space dimension
    state_dim = env.observation_space

    # action space dimension
    action_dim = env.action_space

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma,
                    K_epochs, eps_clip, has_continuous_action_space, action_std, mode=MODE)

    time_step = 0
    i_episode = 0

    # training loop
    while i_episode <= max_training_timesteps:
        current_ep_reward = 0

        state = env.reset()
        state_img = env.state_to_img()

        #if MODE == "mixed":
        #    states = [state['image']]
        #else:
        #    states = [state]
        states = [state_img]

        max_ep_len = env.last_grid_size * 4

        env.steps_threshold = max_ep_len
        for t in range(1, max_ep_len+1):

            # select action with policy
            # action = ppo_agent.select_action(state.view(-1).unsqueeze(0))
            action = ppo_agent.select_action(state)
            state, reward, done = env.step(action)

            state_img = env.state_to_img()
            states.append(state_img)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # printing average reward
            if i_episode % log_freq == 0:
                pass

            # break; if the episode is over
            if done:
                break

        logger.dump_metric(current_ep_reward, i_episode,
                           "train", "episode_reward")

        if i_episode % log_freq == 0:
            t_states = torch.cat(states).unsqueeze(0)
            logger.dump_batch_video(
                t_states, i_episode, "train", "episode_anim")

        i_episode += 1

    env.close()


if __name__ == '__main__':
    train()
