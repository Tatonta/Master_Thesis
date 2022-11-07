import numpy as np
import torch
import torch.nn as nn
from collections import deque
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

# set device to cpu or cuda
device = torch.device(
    'cpu') if torch.cuda.is_available() else torch.device('cuda:0')

if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self, batch_size = 64):
        self.batch_size = batch_size
        self.mem_size = 50000
        self.mem_cntr = 0
        self.actions = np.zeros(self.mem_size)
        self.states_img = np.zeros((self.mem_size, 960))
        self.states_cmp = np.zeros((self.mem_size, 3))
        self.next_states_img = np.zeros((self.mem_size, 960))
        self.next_states_cmp = np.zeros((self.mem_size, 3))
        self.rewards = np.zeros(self.mem_size)
        self.is_terminals = np.zeros(self.mem_size).astype(int)


    def memory(self, state_img, state_cmp, action, reward, next_state_img, next_state_cmp, terminal):
        self.index = self.mem_cntr % self.mem_size

        self.states_img[self.index] = state_img.cpu().numpy().astype(np.float32)
        self.states_cmp[self.index] = state_cmp.cpu().numpy().astype(np.float32)
        self.next_states_img[self.index] = next_state_img.cpu().numpy().astype(np.float32)
        self.next_states_cmp[self.index] = next_state_cmp.cpu().numpy().astype(np.float32)
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.is_terminals[self.index] = terminal

        self.mem_cntr += 1

    def sample_batch(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size)

        state_img_batch = self.states_img[batch]
        state_cmp_batch = self.states_cmp[batch]
        action_batch = self.actions[batch]
        reward_batch = self.rewards[batch]
        next_state_img_batch = self.next_states_img[batch]
        next_state_cmp_batch = self.next_states_cmp[batch]
        is_terminal_batch = self.rewards[batch]

        return state_img_batch, state_cmp_batch, action_batch, reward_batch, next_state_img_batch, next_state_cmp_batch, is_terminal_batch

class View(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self , x):
        b , _ , _ , _ = x.shape
        return x.view(b , -1)


class Network(nn.Module):
    def __init__(self , state_dim , action_dim , mode):
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

        self.actor = nn.Linear(out_size , action_dim)

    def forward(self , x):
        if self.mode == "compass":
            return self.actor(self.mlp(x))
        elif self.mode == "image":
            return self.actor(self.cnn(x))
        else:
            c_feat = self.mlp(x['compass'])
            i_feat = self.cnn(x['image'])
            return self.actor(torch.cat((c_feat, i_feat), 1))


class ActorCritic(nn.Module):
    def __init__(self , state_dim , action_dim , mode="image" , **kwargs):
        super(ActorCritic , self).__init__()

        assert mode in ["compass" , "image" ,
                        "mixed"] , "Attention: state_mode must be one in [compass, image, mixed]"
        self.mode = mode

        conv = True
        # actor
        self.actor = Network(state_dim, action_dim, mode)

        # critic
        self.critic = Network(state_dim, action_dim, mode)
        self.critic.load_state_dict(self.actor.state_dict())

    def forward(self):
        raise NotImplementedError

    def act(self, state, EPS):
        with torch.no_grad():
            Qp = self.actor(state)
        Q, A = torch.max(Qp , axis = -1)
        A = A.item() if torch.rand(1) > EPS else torch.randint(0 , 5 , (1 ,)).item()
        return A

    def evaluate(self , state , action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs , state_values , dist_entropy


class DQN:
    def __init__(self , state_dim , action_dim , lr_actor , lr_critic , gamma , K_epochs , EPS, EPS_END, EPS_DECAY, EPS_START = 1, mode="image"):

        self.mode = mode

        self.gamma = gamma
        self.K_epochs = K_epochs
        self.EPS = EPS
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.EPS_START = EPS_START
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim , action_dim , mode = mode).to(device)
        # self.policy.load_state_dict(torch.load("D:/Logs/rl_navigation/PPO_run_0.pth"))
        self.optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr = lr_actor)
        self.smoothloss = nn.SmoothL1Loss()

    def select_action(self , state):

        action = self.policy.act(state, self.EPS)
        return action

    def store_transition(self, state_img, state_cmp, action, reward, next_state_img, next_state_cmp, terminal):
        self.buffer.memory(state_img, state_cmp, action, reward, next_state_img, next_state_cmp, terminal)

    def update(self, i_episode):

        # Monte Carlo estimate of returns
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            old_states_img , old_states_cmp , actions , rewards , next_states_img , next_states_cmp , terminals = self.buffer.sample_batch()
            # convert list to tensor
            old_states_img = torch.from_numpy(old_states_img).float().squeeze(1).to(device)
            old_states_cmp = torch.from_numpy(old_states_cmp).float().squeeze(1).to(device)
            next_states_img = torch.from_numpy(next_states_img).float().squeeze(1).to(device)
            next_states_cmp = torch.from_numpy(next_states_cmp).float().squeeze(1).to(device)
            next_states = {'image': next_states_img , 'compass': next_states_cmp}
            old_states = {'image': old_states_img , 'compass': old_states_cmp}
            terminals = torch.from_numpy(terminals).to(device)

            # old_actions = torch.squeeze(torch.stack(
            #     self.buffer.actions , dim = 0)).detach().to(device)
            rewards_t = torch.from_numpy(rewards).float().to(device)
            # Evaluating old actions and values
            # logprobs , state_values , dist_entropy = self.policy.evaluate(
            #     old_states , old_actions)
            Q_values = self.policy.actor(old_states)
            # match state_values tensor dimensions with rewards tensor
            # state_values = torch.squeeze(state_values)
            bestAction , _ = torch.max(Q_values, axis = 1)
            # Finding the ratio (pi_theta / pi_theta__old)
            # ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            with torch.no_grad():
                Q_values_next = self.policy.critic(next_states)
            bestNextAction , _ = torch.max(Q_values_next , axis = 1)
            target = rewards_t + self.gamma * bestNextAction * (1-terminals)

            # final loss of clipped objective PPO
            loss = self.smoothloss(bestAction, target)

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy.critic.load_state_dict(self.policy.actor.state_dict())
        self.EPS = np.interp(i_episode , [0 , self.EPS_DECAY] , [self.EPS_START , self.EPS_END])
        # clear buffer

    def save(self , checkpoint_path):
        torch.save(self.policy_old.state_dict() , checkpoint_path)

    def load(self , checkpoint_path):
        self.policy_old.load_state_dict(torch.load(
            checkpoint_path , map_location = lambda storage , loc: storage))
        self.policy.load_state_dict(torch.load(
            checkpoint_path , map_location = lambda storage , loc: storage))