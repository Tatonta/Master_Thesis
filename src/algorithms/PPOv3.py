import copy

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import Categorical
import numpy as np
import itertools
import scipy
# set device to cpu or cuda

if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


################################## PPO Policy ##################################
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class RolloutBuffer:
    def __init__(self , mem_size = 5000, batch_size = 64, img_size=960, gamma=0.99, gae_lambda = 0.97):
        self.mem_size = mem_size
        self.img_size = img_size
        self.states_img = np.zeros((self.mem_size, img_size)).astype(np.float32)
        self.states_cmp = np.zeros((self.mem_size, 3)).astype(np.float32)
        self.probs = np.zeros(self.mem_size).astype(np.float32)
        self.vals = np.zeros(self.mem_size).astype(np.float32)
        self.actions = np.zeros(self.mem_size).astype(np.float32)
        self.rewards = np.zeros(self.mem_size).astype(np.int)
        self.dones = np.zeros(self.mem_size).astype(np.float32)
        self.counter = 0
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.adv_buf = np.zeros(self.mem_size)

    def generate_batch(self):
        max_mem = self.counter
        if self.counter > self.mem_size:
            max_mem = self.mem_size
        batch = np.random.choice(max_mem-1, self.batch_size)
        deltas = self.rewards[:-1] + self.gamma * self.vals[1:] - self.vals[:-1]
        total_discount_sum = discount_cumsum(deltas , self.gamma * self.gae_lambda)
        self.adv_buf[batch] = total_discount_sum[batch]
        return self.states_img[batch] , self.states_cmp[batch] , self.actions[batch] , self.probs[batch] , self.vals[
            batch] , self.rewards[batch] , self.dones[batch] , self.adv_buf[batch]

    def store_transition(self , state_img, state_cmp , action , probs , vals , reward , done):
        self.idx = self.counter % self.mem_size
        self.states_img[self.idx] = state_img.cpu().numpy().astype(np.float32)
        self.states_cmp[self.idx] = state_cmp.cpu().numpy().astype(np.float32)
        self.actions[self.idx] = action
        self.probs[self.idx] = probs
        self.vals[self.idx] = vals
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.counter += 1

    def clear_memory(self):
        self.states_img = np.zeros((self.mem_size, self.img_size)).astype(np.float32)
        self.states_cmp = np.zeros((self.mem_size, 3)).astype(np.float32)
        self.probs = np.zeros(self.mem_size).astype(np.float32)
        self.vals = np.zeros(self.mem_size).astype(np.float32)
        self.actions = np.zeros(self.mem_size).astype(np.float32)
        self.rewards = np.zeros(self.mem_size).astype(np.int)
        self.dones = np.zeros(self.mem_size).astype(np.float32)

class View(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self , x):
        b , _ , _ , _ = x.shape
        return x.view(b , -1)


class Network(nn.Module):
    def __init__(self , state_dim , action_dim , mode , is_actor=False):
        super(Network , self).__init__()

        self.mode = mode
        if mode == "compass" or mode == "mixed":
            self.mlp = nn.Sequential(
                nn.Linear(state_dim , 64) ,
                nn.Tanh() ,
                nn.Linear(64 , 64) ,
                nn.Tanh()
            )

            out_size = 64

        if mode == "image" or mode == "mixed":
            self.cnn = nn.Sequential(
                nn.Linear(960 , 256) ,
                nn.ReLU() ,

                nn.Linear(256 , 128) ,
                nn.ReLU() ,

                nn.Linear(128 , 64) ,
                nn.ReLU() ,
            )

            out_size = 64

        if mode == "mixed":
            out_size = 64 + 64

        if is_actor:
            self.actor = nn.Sequential(
                nn.Linear(out_size , action_dim) ,
                nn.Softmax(dim = -1))
        else:
            self.actor = nn.Linear(out_size , 1)

    def forward(self , x):
        if self.mode == "compass":
            return self.actor(self.mlp(x))
        elif self.mode == "image":
            return self.actor(self.cnn(x))
        else:
            c_feat = self.mlp(x['compass'])
            i_feat = self.cnn(x['image'])
            return self.actor(torch.cat((c_feat , i_feat) , 1))


class ActorCritic(nn.Module):
    def __init__(self , state_dim , action_dim , mode="image" , **kwargs):
        super(ActorCritic , self).__init__()

        assert mode in ["compass" , "image" ,
                        "mixed"] , "Attention: state_mode must be one in [compass, image, mixed]"
        self.mode = mode

        conv = True
        # actor
        self.actor = Network(state_dim , action_dim , mode , is_actor = True)
        # critic
        self.critic = Network(state_dim , action_dim , mode , is_actor = False)

    def forward(self):
        raise NotImplementedError

    def act(self , state):
        prob_dist = self.actor(state)
        dist = Categorical(prob_dist)
        value = self.critic(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    def evaluate(self , state , action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs , state_values , dist_entropy


class PPO:
    def __init__(self , state_dim , action_dim , lr_actor , lr_critic , gamma , K_epochs , eps_clip , mode="image"):

        self.mode = mode

        self.gamma = gamma
        self.gae_lambda = 0.95
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer(batch_size = 64)

        self.policy = ActorCritic(state_dim, action_dim, mode=mode).to(device)
        #self.policy.load_state_dict(torch.load("D:/Logs/rl_navigation/PPO_run_0.pth"))
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, mode=mode).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self , state):
        actions, log_prob, value = self.policy_old.act(state)
        return actions, log_prob, value

    def store(self, s_img, s_cmp, a, prob, val, r, done):
        self.buffer.store_transition(s_img,s_cmp, a, prob, val, r, done)

    def update(self):
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            state_img_arr, state_cmp_arr , action_arr , old_prob_arr , vals_arr , \
            reward_arr , dones_arr , adv_buf = \
                self.buffer.generate_batch()


            advantages = torch.tensor(adv_buf, dtype = torch.float).to(device)

            values = torch.tensor(vals_arr, dtype = torch.float).to(device)

            states_img = torch.tensor(state_img_arr, dtype = torch.float).squeeze(1).to(device)
            states_cmp = torch.tensor(state_cmp_arr, dtype = torch.float).squeeze(1).to(device)
            old_logprobs = torch.tensor(old_prob_arr, dtype = torch.float).to(device)
            old_actions = torch.tensor(action_arr, dtype = torch.float).to(device)

            old_states = {
                'image': states_img,
                'compass': states_cmp
            }
            logprobs , state_values , dist_entropy = self.policy.evaluate(
                old_states , old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)
            dist_entropy = dist_entropy.type(torch.float).mean()
            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios , 1 - self.eps_clip ,
                                1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1 , surr2).mean()

            returns = advantages + values
            critic_loss = self.MseLoss(returns, state_values)

            total_loss = loss + 0.5*critic_loss-0.01*dist_entropy
            # take gradient step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear_memory()