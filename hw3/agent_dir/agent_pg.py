import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from agent_dir.agent import Agent
from agent_dir.net import PolicyNet, ActorCritic
from agent_dir.memory import Memory
from environment import Environment

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
class AgentPG(Agent):
    def __init__(self, env, args):
        self.args = args
        self.env = env
        if not hasattr(self.args, 'env_name')\
           or self.args.env_name is None:
            self.args.env_name = "LunarLander"
        try:
            import matplotlib.pyplot as plt
            self.plot = True
        except ImportError:
            self.plot = False

        env.seed(args.seed)
        torch.manual_seed(args.seed)

        self.model = PolicyNet(state_dim=self.env.observation_space.shape[0],
                               action_num=self.env.action_space.n,
                               hidden_dim=64)
        if args.test_pg:
            self.load('model/pg.cpt')

        # discounted reward
        self.gamma = 0.99
        
        # training hyperparameters
        self.num_episodes = 100000 # total training episodes (actually too large...)
        self.display_freq = 10 # frequency to display training progress
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        
        # saved rewards and actions
        self.rewards, self.saved_actions = [], []
        self.saved_log_probs = []

        self.eps = np.finfo(np.float32).eps.item()
    
    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        print('load model from', load_path)
        self.model.load_state_dict(torch.load(load_path))

    def init_game_setting(self):
        self.rewards, self.saved_actions = [], []

    def make_action(self, state, test=False):
        # TODO:
        # Use your model to output distribution over actions and sample from it.
        # HINT: google torch.distributions.Categorical

        state = torch.from_numpy(state).type(torch.FloatTensor).view(1, -1)
        probs = self.model(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def update(self):
        # TODO:
        # discount your saved reward
        R = 0
        rewards = []
        policy_loss = []

        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        # Scale rewards
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)

        # TODO:
        # compute loss
        for log_prob, R in zip(self.saved_log_probs, rewards):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        #policy_loss = torch.cat(policy_loss).sum()
        policy_loss = torch.cat(policy_loss).mean()
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]

    def train(self):
        avg_reward = None # moving average of reward
        last_k_rewards = []
        last_k_avgs = []
        k = 100
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            for _ in range(int(1e4)):
                action = self.make_action(state)
                state, reward, done, _ = self.env.step(action)
                
                self.saved_actions.append(action)
                self.rewards.append(reward)
                
                if done: break

            # for logging 
            last_reward = np.sum(self.rewards)
            avg_reward = last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1

            # for plotting
            last_k_rewards.append(last_reward)
            last_k_rewards = last_k_rewards[-k:]
            last_k_avg = np.array(last_k_rewards).mean()
            last_k_avgs.append(last_k_avg)
            
            # update model
            self.update()

            if epoch % self.display_freq == 0:
                print(f"Epochs: {epoch}/{self.num_episodes} | Reward Avg: {avg_reward} | Last {k} Avg: {last_k_avg}")

            if epoch % k == 0:
                self.save(self.args.model_path)
                np.save(self.args.model_path + '.npy', np.array(last_k_avgs, dtype=np.float))
                if self.plot:
                    import matplotlib.pyplot as plt
                    plt.plot(np.arange(len(last_k_avgs), dtype=np.int)[:], last_k_avgs[:])
                    plt.title(f"PG Learning Curve on {self.args.env_name}")
                    plt.ylabel(f"Average Rewards of last {k} episodes")
                    plt.xlabel("Episode")
                    plt.savefig(self.args.model_path + '.png')
                    print(f"Learning curve saved at {self.args.model_path + '.png'}")

            if last_k_avg > self.args.target_score:
                self.save(self.args.model_path)
                break


class AgentPPO_basic(Agent):                          
    def __init__(self, env, args):
        self.args = args
        self.env = env
        try:
            import matplotlib.pyplot as plt
            self.plot = True
        except ImportError:
            self.plot = False

        env.seed(args.seed)
        torch.manual_seed(args.seed)
        
        action_dim = 4
        self.num_episodes = 100000
        self.display_freq = 10
        self.betas = (0.9, 0.999)
        
        self.memory = Memory()
        self.MseLoss = nn.MSELoss()
        self.policy = ActorCritic(self.env.observation_space.shape[0],
                                  action_dim, 
                                  self.args.n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=self.args.ppo_lr, 
                                          betas=self.betas)
        self.policy_old = ActorCritic(self.env.observation_space.shape[0], 
                                      action_dim,
                                      self.args.n_latent_var).to(device)
        if args.test_pg:
            self.load(self.args.model_path)
        
    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.policy.state_dict(), save_path)
    
    def load(self, load_path):
        print('load model from', load_path)
        ckpt = torch.load(load_path)
        self.policy.load_state_dict(ckpt)
        self.policy_old.load_state_dict(ckpt)
    
    def update(self):   
        rewards = []
        discounted_reward = 0
        for reward in reversed(self.memory.rewards):
            discounted_reward = reward + (self.args.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        old_states = torch.stack(self.memory.states).to(device).detach()
        old_actions = torch.stack(self.memory.actions).to(device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(device).detach()
        
        for _ in range(self.args.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.args.eps_clip, 1+self.args.eps_clip) * advantages
            loss = (-1) * torch.min(surr1, surr2) +\
                    0.5 * self.MseLoss(state_values, rewards) -\
                    0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.sum().backward()
            self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())

    def train(self):
        last_k_rewards = []
        last_k_avgs = []
        k = 100
        
        time_step = 0
        for epoch in range(self.num_episodes):
            state = self.env.reset()
            episode_reward = 0
            for _ in range(int(2e4)):
                time_step += 1
                action = self.policy_old.act(state, self.memory)
                state, reward, done, _ = self.env.step(action)

                episode_reward += reward
                self.memory.rewards.append(reward)
                
                if time_step % self.args.update_timestep == 0:
                    self.update()
                    self.memory.clear_memory()
                    time_step = 0

                if done: 
                    break

            # for plotting
            last_k_rewards.append(episode_reward)
            last_k_rewards = last_k_rewards[-k:]
            last_k_avg = np.array(last_k_rewards).mean()
            last_k_avgs.append(last_k_avg)
            
            # update model
            self.update()

            if epoch % self.display_freq == 0:
                print(f"Epochs: {epoch}/{self.num_episodes} "
                      f"| Last {k} Avg: {last_k_avg}")

            if epoch % k == 0:
                self.save(self.args.model_path)
                np.save(self.args.model_path + '.npy', np.array(last_k_avgs, dtype=np.float))
                if self.plot:
                    import matplotlib.pyplot as plt
                    plt.plot(np.arange(len(last_k_avgs), dtype=np.int)[:], last_k_avgs[:])
                    plt.title(f"PPO Learning Curve on {self.args.env_name}")
                    plt.ylabel(f"Average Rewards of last {k} episodes")
                    plt.xlabel("Episode")
                    plt.savefig(self.args.model_path + '.png')
                    print(f"Learning curve saved at {self.args.model_path + '.png'}")
