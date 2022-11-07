from pathlib import Path

####### initialize environment hyperparameters ######

env_name = "GridEnvironment-Mixed"

exp_path = Path("D:/Logs/rl_navigation")

has_continuous_action_space = False  # continuous action space; else discrete

max_ep_len = 100                   # max timesteps in one episode
# break training loop if timeteps > max_training_timesteps
max_training_timesteps = int(230000)

# log on tensorboard (in num episodes)
log_freq = 100
# save model frequency (in num timesteps)
save_model_freq = int(1e5)

# starting std for action distribution (Multivariate Normal)
action_std = 0.6
# linearly decay action_std (action_std = action_std - action_std_decay_rate)
action_std_decay_rate = 0.05
# minimum action_std (stop decay after action_std <= min_action_std)
min_action_std = 0.1
# action_std decay frequency (in num timesteps)
action_std_decay_freq = int(2.5e5)

EPS = 1
EPS_END = 0.05
EPS_DECAY = 2500

#####################################################

################ PPO hyperparameters ################

update_timestep = max_ep_len * 2     # update policy every n timesteps
K_epochs = 10               # update policy for K epochs in one PPO update

eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

random_seed = 0         # set random seed if required (0 = no random seed)

#####################################################


'''
    path = Path("D:/Logs/rl_navigation")
    logger = TBSaver(base_output_dir=path, title="toy_env_test", args={})

    env = GridEnvironment(visual_state=True, steps_threshold=0, visual_debug=False, debug_rate=100)

    state = env.reset()
    states = [state]

    for t in range(1, 100):
        action = randrange(0, 4)
        state, reward, done = env.step(action)

        states += [state]

        # break; if the episode is over
        if done:
            break

    states = torch.cat(states).unsqueeze(0)
    logger.dump_batch_video(states, 0, "train", "episode")
    n, t, c, h, w = states.shape
    logger.dump_batch_image(states.view(n*t, c, h, w), 0, "train", "img_episode")
    print("End")

'''