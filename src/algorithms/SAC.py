import copy

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import Categorical
import numpy as np
import itertools

# set device to cpu or cuda

if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self , max_size , input_shape , n_actions, img_size):
        self.mem_size = max_size
        self.counter = 0
        self.states_img_memory = np.zeros((self.mem_size, img_size))
        self.states_cmp_memory = np.zeros((self.mem_size, input_shape))
        self.actions_memory = np.zeros(self.mem_size).astype(np.int64)
        self.new_states_img_memory = np.zeros((self.mem_size, img_size))
        self.new_states_cmp_memory = np.zeros((self.mem_size, input_shape))
        self.rewards_memory = np.zeros(self.mem_size)
        self.dones = np.zeros(self.mem_size).astype(np.int64)

    def store_transition(self , state_img , state_cmp, action , reward , new_state_img, new_state_cmp , done):
        self.index = self.counter % self.mem_size
        self.states_img_memory[self.index] = state_img
        self.states_cmp_memory[self.index] = state_cmp
        self.actions_memory[self.index] = action
        self.rewards_memory[self.index] = reward
        self.new_states_img_memory[self.index] = new_state_img
        self.new_states_cmp_memory[self.index] = new_state_cmp
        self.dones[self.index] = done
        self.counter += 1

    def sample_buffer(self , batch_size):
        max_mem = self.counter
        if self.counter > self.mem_size:
            max_mem = self.mem_size
        batch = np.random.choice(max_mem , batch_size)

        states_img = self.states_img_memory[batch]
        states_cmp = self.states_cmp_memory[batch]
        rewards = self.rewards_memory[batch]
        new_states_img = self.new_states_img_memory[batch]
        new_states_cmp = self.new_states_cmp_memory[batch]
        actions = self.actions_memory[batch]
        dones = self.dones[batch]

        return states_img , states_cmp, rewards , actions , new_states_img, new_states_cmp , dones


class View(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self , x):
        b , _ , _ , _ = x.shape
        return x.view(b , -1)


class ActorNet(nn.Module):
    def __init__(self , state_dim , action_dim, mode):
        super(ActorNet , self).__init__()
        self.mode = mode
        self.reparam_noise = 1e-6
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
        self.action = nn.Sequential(
            nn.Linear(out_size , action_dim),
            nn.Softmax(dim=-1)
        )


    def forward(self, x):
        if self.mode == "compass":
            action_dist = self.action(self.mlp(x['compass']))
            log_probs = torch.log(action_dist+self.reparam_noise)
            return action_dist, log_probs
        elif self.mode == "image":
            action_dist = self.action(self.cnn(x['image']))
            log_probs = torch.log(action_dist+self.reparam_noise)
            return action_dist, log_probs
        else:
            c_feat = self.mlp(x['compass'])
            i_feat = self.cnn(x['image'])
            action_dist = self.action(torch.cat((c_feat , i_feat) , dim=-1))
            log_probs = torch.log(action_dist+self.reparam_noise)
            return action_dist, log_probs

    def act(self, state):
        action_dist, log_probs = self.forward(state)
        m = Categorical(action_dist)
        action = m.sample()
        return action.item()

class Q_Net(nn.Module):
    def __init__(self , state_dim , action_dim , mode):
        super(Q_Net , self).__init__()

        self.mode = mode
        if mode == "compass" or mode == "mixed":
            self.mlp = nn.Sequential(
                nn.Linear(state_dim, 64) ,
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

        self.Q = nn.Linear(out_size, action_dim)

    def forward(self, x):
        if self.mode == "compass":
            return self.Q(self.mlp(x['compass']))
        elif self.mode == "image":
            return self.Q(self.cnn(x['image']))
        else:
            c_feat = self.mlp(x['compass'])
            i_feat = self.cnn(x['image'])
            return self.Q(torch.cat((c_feat, i_feat), dim=-1))



class SAC:
    def __init__(self , state_dim , action_dim , lr_actor , lr_critic , gamma , K_epochs ,max_action, batch_size = 256, mode="image"):

        assert mode in ["compass" , "image" ,
                        "mixed"] , "Attention: state_mode must be one in [compass, image, mixed]"
        self.mode = mode
        self.tau = 0.005
        self.batch_size = batch_size
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.max_action = max_action
        self.ALPHA_INITIAL = 1
        conv = True
        # actor
        self.actor = ActorNet(state_dim , action_dim , mode).to(device)

        # critic
        self.Q1 = Q_Net(state_dim , action_dim , mode).to(device)
        self.Q2 = Q_Net(state_dim, action_dim, mode).to(device)
        self.Q1_targ = copy.deepcopy(self.Q1)
        self.Q2_targ = copy.deepcopy(self.Q2)

        self.MSE_Criterion = nn.MSELoss()
        self.buffer = RolloutBuffer(100000, state_dim, action_dim, 960)
        # self.policy.load_state_dict(torch.load("D:/Logs/rl_navigation/PPO_run_0.pth"))
        #Optimizers of the networks
        self.q1_optim = torch.optim.Adam(self.Q1.parameters() , lr = 1e-4)
        self.q2_optim = torch.optim.Adam(self.Q2.parameters() , lr = 1e-4)
        self.actor_optim = torch.optim.Adam(self.actor.parameters() , lr = 1e-4)
        self.q_targ_p = itertools.chain(self.Q1_targ.parameters() , self.Q2_targ.parameters())
        for parameter in self.q_targ_p:
            parameter.requires_grad = False

        self.target_entropy = 0.98 * -np.log(1 / action_dim)
        self.log_alpha = torch.tensor(np.log(self.ALPHA_INITIAL) , requires_grad = True)
        self.alpha = self.log_alpha
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha] , lr = 1e-4)

    def select_action(self , state):
        actions = self.actor.act(state)
        return actions

    def update_network_parameters(self):
        for target_param, param in zip(self.Q1_targ.parameters(), self.Q1.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)

        for target_param, param in zip(self.Q2_targ.parameters(), self.Q2.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)

    def store(self, s_img, s_cmp, a, r, ns_img, ns_cmp, done):
        self.buffer.store_transition(s_img,s_cmp, a,r,ns_img, ns_cmp,done)

    def update(self):

        # Monte Carlo estimate of returns
        if self.buffer.counter < self.batch_size:
            return
        states_img, states_cmp, rewards, actions, new_states_img, new_states_cmp, dones = self.buffer.sample_buffer(self.batch_size)

        if self.mode == "mixed":
            states_img_t = torch.from_numpy(states_img).float().to(device)
            states_cmp_t = torch.from_numpy(states_cmp).float().to(device)
            rewards_t = torch.from_numpy(rewards).float().to(device)
            actions_t = torch.from_numpy(actions).to(device)
            new_states_img = torch.from_numpy(new_states_img).float().to(device)
            new_states_cmp = torch.from_numpy(new_states_cmp).float().to(device)
            dones_t = torch.from_numpy(dones).to(device)
        # old_actions = torch.squeeze(torch.stack(
        #     self.buffer.actions , dim = 0)).detach().to(device)
        else:
            states_cmp_t = torch.from_numpy(states_cmp).float().to(device)
            rewards_t = torch.from_numpy(rewards).float().to(device)
            actions_t = torch.from_numpy(actions).to(device)
            new_states_cmp = torch.from_numpy(new_states_cmp).float().to(device)
            dones_t = torch.from_numpy(dones).to(device)
        states_t = {
            'image': states_img_t,
            'compass': states_cmp_t
        }
        new_states_t = {
            "image": new_states_img,
            "compass": new_states_cmp
        }
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            self.q1_optim.zero_grad()
            self.q2_optim.zero_grad()
            self.actor_optim.zero_grad()
            self.alpha_optimizer.zero_grad()
            q1 = self.Q1(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
            q2 = self.Q2(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                actions_probs , log_probs = self.actor(new_states_t)
                q1t_value = self.Q1_targ(new_states_t)
                q2t_value = self.Q2_targ(new_states_t)
                predicted_new_q = torch.min(q1t_value , q2t_value)
                backup = rewards_t + self.gamma * (1 - dones_t) * (
                            actions_probs * (predicted_new_q - self.alpha * log_probs)).sum(dim = -1)

            loss_q1 = self.MSE_Criterion(q1 , backup)
            loss_q2 = self.MSE_Criterion(q2 , backup)
            loss_q1.backward()
            loss_q2.backward()
            self.q1_optim.step()
            self.q2_optim.step()

            # LOSS FOR ACTOR
            action_dist , log_probs = self.actor(states_t)
            q1_new_policy = self.Q1(states_t)
            q2_new_policy = self.Q2(states_t)
            qp_min = torch.min(q1_new_policy , q2_new_policy)
            pi_loss = (action_dist * (self.alpha * log_probs - qp_min)).sum(dim = -1).mean()
            pi_loss.backward()
            self.actor_optim.step()

            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
            # Updating Policy

            # Finally, update target networks by polyak averaging.
            # with torch.no_grad():
            #     for p_targ, p in zip(self.V_targ_params, self.V_params):
            #         # NB: We use an in-place operations "mul_", "add_" to update target
            #         # params, as opposed to "mul" and "add", which would make new tensors.
            #         p_targ.data.mul_(self.polyak)
            #         p_targ.data.add_((1 - self.polyak) * p.data)

            self.update_network_parameters()

    def save(self , checkpoint_path):
        torch.save(self.actor.state_dict() , checkpoint_path+"actor")
        torch.save(self.Q1.state_dict(), checkpoint_path+"Q1")
        torch.save(self.Q2.state_dict(), checkpoint_path+"Q2")
        torch.save(self.Q1_targ.state_dict(), checkpoint_path+"Q1_targ")
        torch.save(self.Q2_targ.state_dict(), checkpoint_path+"Q2_targ")

    def load(self , checkpoint_path):
        self.actor.load_state_dict(torch.load(
            checkpoint_path+"actor" , map_location = lambda storage , loc: storage))
        self.Q1.load_state_dict(torch.load(
            checkpoint_path+"Q1" , map_location = lambda storage , loc: storage))
        self.Q2.load_state_dict(torch.load(
            checkpoint_path+"Q2" , map_location = lambda storage , loc: storage))
        self.Q1_targ.load_state_dict(torch.load(
            checkpoint_path+"Q1_targ" , map_location = lambda storage , loc: storage))
        self.Q2_targ.load_state_dict(torch.load(
            checkpoint_path+"Q2_targ" , map_location = lambda storage , loc: storage))