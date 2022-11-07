import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class View(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, _, _, _ = x.shape
        return x.view(b, -1)


class Network(nn.Module):
    def __init__(self, state_dim, action_dim, mode, is_actor=False):
        super(Network, self).__init__()

        self.mode = mode
        if mode == "compass" or mode == "mixed":
            self.mlp = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh()
            )

            out_size = 64

        if mode == "image" or mode == "mixed":
            self.cnn = nn.Sequential(
                nn.Linear(960, 256),
                nn.ReLU(),

                nn.Linear(256, 128),
                nn.ReLU(),
                
                nn.Linear(128, 64),
                nn.ReLU(),
            )

            out_size = 64

        if mode == "mixed":
            out_size = 64 + 64

        if is_actor:
            self.actor = nn.Sequential(
                nn.Linear(out_size, action_dim), 
                nn.Softmax(dim=-1))
        else:
            self.actor = nn.Linear(out_size, 1)

    def forward(self, x):
        if self.mode == "compass":
            return self.actor(self.mlp(x))
        elif self.mode == "image":
            return self.actor(self.cnn(x))
        else:
            c_feat = self.mlp(x['compass'])
            i_feat = self.cnn(x['image'])
            return self.actor(torch.cat((c_feat, i_feat), 1))


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, mode="image", **kwargs):
        super(ActorCritic, self).__init__()

        assert mode in ["compass", "image",
                        "mixed"], "Attention: state_mode must be one in [compass, image, mixed]"
        self.mode = mode

        conv = True
        # actor
        self.actor = Network(state_dim, action_dim, mode, is_actor=True)

        # critic
        self.critic = Network(state_dim, action_dim, mode, is_actor=False)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, mode="image"):

        self.mode = mode

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, mode=mode).to(device)
        #self.policy.load_state_dict(torch.load("D:/Logs/rl_navigation/PPO_run_0.pth"))
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, mode=mode).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):

        with torch.no_grad():
            if self.mode == 'mixed':
                state = {
                    'image': torch.FloatTensor(state["image"]).to(device),
                    'compass': torch.FloatTensor(state["compass"].unsqueeze(0)).to(device)
                }
            else:
                state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item()

    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        if self.mode == 'mixed':
            old_states_img = []
            old_states_cmp = []
            for dict in self.buffer.states:
                old_states_img.append(dict['image'])
                old_states_cmp.append(dict['compass'])
            old_states_img = torch.squeeze(torch.stack(
                old_states_img, dim=0)).detach().to(device)
            old_states_cmp = torch.squeeze(torch.stack(
                old_states_cmp, dim=0)).detach().to(device)
            old_states = {'image': old_states_img, 'compass': old_states_cmp}
        else:
            old_states = torch.squeeze(torch.stack(
                self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(
            self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(
            self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage))
