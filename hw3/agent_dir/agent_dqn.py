import os
import random
from collections import namedtuple
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.distributions.categorical import Categorical

from agent_dir.agent import Agent
from agent_dir import net
from agent_dir.memory import ReplayMemory
from environment import Environment


use_cuda = torch.cuda.is_available()
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
episodes_done_num = 0


class AgentDQN(Agent):
    def __init__(self, env, args):
        self.args = args
        try:
            import matplotlib.pyplot as plt
            self.plot = True
        except ImportError:
            self.plot = False

        self.env = env
        self.input_channels = 4
        self.num_actions = self.env.action_space.n
        # TODO:
        # Initialize your replay buffer
        self.replay_buffer = ReplayMemory(self.args.buffer_size)

        # build target, online network
        self.target_net = getattr(net, self.args.dqn_net)(self.input_channels, self.num_actions)
        self.target_net = self.target_net.cuda() if use_cuda else self.target_net
        self.online_net = getattr(net, self.args.dqn_net)(self.input_channels, self.num_actions)
        self.online_net = self.online_net.cuda() if use_cuda else self.online_net

        if args.test_dqn:
            self.load('model/dqn')
        
        # discounted reward
        self.GAMMA = self.args.GAMMA
        
        # training hyperparameters
        self.train_freq = self.args.train_freq
        self.learning_start = self.args.learning_start
        self.batch_size = self.args.dqn_batch_size
        self.num_timesteps = self.args.num_timesteps
        self.display_freq = self.args.display_freq
        self.save_freq = self.args.save_freq
        self.target_update_freq = self.args.target_update_freq

        # optimizer
        self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=self.args.dqn_lr)

        self.steps = 0  # num. of passed steps. this may be useful in controlling exploration

    def save(self, save_path):
        print('save model to', save_path)
        torch.save(self.online_net.state_dict(), save_path + '_online.cpt')
        torch.save(self.target_net.state_dict(), save_path + '_target.cpt')

    def load(self, load_path):
        print('load model from', load_path)
        if use_cuda:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt'))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt'))
        else:
            self.online_net.load_state_dict(torch.load(load_path + '_online.cpt', map_location=lambda storage, loc: storage))
            self.target_net.load_state_dict(torch.load(load_path + '_target.cpt', map_location=lambda storage, loc: storage))

    def init_game_setting(self):
        # we don't need init_game_setting in DQN
        pass
    
    def make_action(self, state, test=False):
        if test:
            state = torch.tensor(state, device='cuda' if use_cuda else 'cpu').permute(2, 0, 1).unsqueeze(0)

        if self.args.exploration_method.startswith('greedy'):
            with torch.no_grad():
                actions = torch.softmax(self.online_net(state), 1).max(1)[1].view(-1, 1)
            if test:
                return actions.item()
            return actions

        elif self.args.exploration_method.startswith('epsilon'):
            # TODO:
            # At first, you decide whether you want to explore the environemnt
            sample = random.random()

            if self.args.exploration_method.startswith('epsilon_exp'):
                global episodes_done_num
                EPS_START = 0.9
                EPS_END = 0.1
                EPS_DECAY = 200
                eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * episodes_done_num / EPS_DECAY)
            else:
                eps_threshold = .1

            # TODO:
            # if explore, you randomly samples one action
            # else, use your model to predict action
            if sample > eps_threshold or test:
                with torch.no_grad():
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    if test:
                        return self.online_net(state).max(1)[1].view(1, 1).item()
                    return self.online_net(state).max(1)[1].view(1, 1)
            else:
                return torch.tensor([[random.randrange(self.num_actions)]],
                                    device='cuda' if use_cuda else 'cpu',
                                    dtype=torch.long)

        elif self.args.exploration_method.startswith('boltzmann'):
            with torch.no_grad():
                probs = torch.softmax(self.online_net(state) / self.args.boltzmann_temperature, 1)
            m = Categorical(probs)
            action = m.sample().view(-1, 1)
            if test:
                return action.item()
            return action

        elif self.args.exploration_method.startswith('thompson'):
            with torch.no_grad():
                if test:
                    probs = torch.softmax(self.online_net.forward(state, dropout_rate=0, thompson=False), 1)
                else:
                    probs = torch.softmax(self.online_net.forward(state, dropout_rate=0.3, thompson=True), 1)
            actions = probs.max(1)[1].view(-1, 1)
            if test:
                return actions.item()
            return actions

        else:
            raise ValueError("Unknown exploration method")

    def update(self):
        # TODO:
        # To update model, we sample some stored experiences as training examples.
        batch = Transition(*zip(*self.replay_buffer.sample(self.batch_size)))

        non_final_mask = torch.tensor(tuple(map(lambda x: x is not None, batch.next_state)),
                                      device='cuda' if use_cuda else 'cpu',
                                      dtype=torch.uint8)
        non_final_next_states = torch.cat([next_state for next_state in batch.next_state if next_state is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.tensor(batch.reward).to('cuda' if use_cuda else 'cpu')

        # TODO:
        # Compute Q(s_t, a) with your model.
        if self.args.exploration_method.startswith('thompson'):
            state_action_values = self.online_net.forward(state_batch,
                                                          dropout_rate=0.3,
                                                          thompson=True).gather(1, action_batch)
        else:
            state_action_values = self.online_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            # TODO:
            # Compute Q(s_{t+1}, a) for all next states.
            # Since we do not want to backprop through the expected action values,
            # use torch.no_grad() to stop the gradient from Q(s_{t+1}, a)
            next_state_values = torch.zeros(self.batch_size, device='cuda' if use_cuda else 'cpu')
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach() if not self.args.DoubleDQN else\
                                                self.target_net(non_final_next_states).gather(1, self.online_net(non_final_next_states).max(1)[1].unsqueeze(-1)).squeeze()

        # TODO:
        # Compute the expected Q values: rewards + gamma * max(Q(s_{t+1}, a))
        # You should carefully deal with gamma * max(Q(s_{t+1}, a)) when it is the terminal state.
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # TODO:
        # Compute temporal difference loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self):
        global episodes_done_num
        episodes_done_num = 0  # passed episodes
        total_reward = 0  # compute average reward
        loss = 0
        # for learning curve
        episode_reward = 0
        last_k_reward = []
        last_k_avgs = []
        k = 500
        while True:
            state = self.env.reset()
            # State: (80,80,4) --> (1,4,80,80)
            state = torch.from_numpy(state).permute(2,0,1).unsqueeze(0)
            state = state.cuda() if use_cuda else state
            
            done = False
            while not done:
                # select and perform action
                action = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action[0, 0].data.item())
                total_reward += reward
                episode_reward += reward

                # process new state
                next_state = torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0)
                next_state = next_state.cuda() if use_cuda else next_state
                if done:
                    next_state = None

                # TODO:
                # store the transition in memory
                self.replay_buffer.push(state, action, next_state, reward)

                # move to the next state
                state = next_state

                # Perform one step of the optimization
                if self.steps > self.learning_start and self.steps % self.train_freq == 0:
                    loss = self.update()

                # update target network
                if self.steps > self.learning_start and self.steps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                # save the model
                if self.steps % self.save_freq == 0:
                    self.save(self.args.model_path)
                    np.save(self.args.model_path+'.npy', np.array(last_k_avgs, dtype=np.float))
                    if self.plot:
                        import matplotlib.pyplot as plt
                        plt.plot(np.arange(len(last_k_avgs), dtype=np.int)[k:], last_k_avgs[k:])
                        plt.title(f"DQN Learning Curve on {self.args.env_name}")
                        plt.ylabel(f"Clipped Averaged Rewards of last {k} episodes")
                        plt.xlabel("Episode")
                        plt.savefig(self.args.model_path)
                        print(f"Learning curve saved at {self.args.model_path}")

                self.steps += 1

            # learning curve
            last_k_reward.append(episode_reward)
            last_k_reward = last_k_reward[-k:]
            last_k_avgs.append(np.mean(last_k_reward))
            episode_reward = 0

            if episodes_done_num % self.display_freq == 0:
                print('Episode: %d | Steps: %d/%d | Avg reward: %f | loss: %f '%
                        (episodes_done_num, self.steps, self.num_timesteps, total_reward / self.display_freq, loss))
                total_reward = 0

            episodes_done_num += 1
            if self.steps > self.num_timesteps:
                break
        self.save(self.args.model_path)
