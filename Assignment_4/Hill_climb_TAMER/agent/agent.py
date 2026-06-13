"""
    Copyright 2024 by Michał Stolarz <michal.stolarz@h-brs.de>, Ritwik Sinha <ritwik.sinha@smail.inf.h-brs.de>

    This file is part of pyirl,
    and is based on: https://github.com/benibienz/TAMER.
    It contains the class of an interactive agent.

    pyirl is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    pyirl is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.
    You should have received a copy of the GNU Affero General Public License
    along with pyirl. If not, see <http://www.gnu.org/licenses/>.
"""

import datetime as dt
import os
import pickle
import time
import uuid
from collections import deque
from itertools import count
from sys import stdout
from csv import DictWriter
from models.tamer import Tamer
from models.utils.sgd_function_approximator import SGDFunctionApproximator
from models.qlearning import Qlearning
import numpy as np
import config as cfg


class Agent:
    """
    Interactive agent.
    """
    def __init__(
        self,
        env,
        model,
        num_episodes,
        credit_assignment,
        discount_factor=1,  # only affects Q-learning
        epsilon=0, # only affects Q-learning
        min_eps=0,  # minimum value for epsilon after annealing
        interactive=True,  # set to false for normal Q-learning
        ts_len=0.2,  # length of timestep for training
        human_answer_interval=(0.2, 0.6),
        output_dir=None,
        model_file_to_save=None,
        model_file_to_load=None  # filename of pretrained model
    ):
        self.interactive = interactive
        self.human_answer_interval = human_answer_interval
        self.ts_len = ts_len
        self.env = env
        self.model = model
        self.credit_assignment = credit_assignment
        self.uuid = uuid.uuid4()
        self.output_dir = output_dir
        self.model_file_to_save = model_file_to_save
        self.model_file_to_load = model_file_to_load
        self.supported_models = ('tamer', 'qlearning')

        # init model
        if model_file_to_load is not None:
            print(f'Loaded pretrained model: {model_file_to_load}')
            self.load_model(filename=model_file_to_load)
        else:
            if interactive:
                if self.model in self.supported_models: # init H function
                    if self.model == self.supported_models[0]:
                        self.H = Tamer(env,
                                       credit_assignment=self.credit_assignment,
                                       human_answer_interval=human_answer_interval)
                    elif self.model == self.supported_models[1]:
                        self.H = Qlearning(env)
                else:
                    raise NotImplementedError('Model type not implemented')
            else:  # optionally run as standard Q Learning
                self.Q = SGDFunctionApproximator(env)  # init Q function

        # hyperparameters
        self.discount_factor = discount_factor
        self.epsilon = epsilon if not interactive else 0
        self.num_episodes = num_episodes
        self.min_eps = min_eps
        self.transitions_buffer = deque(maxlen=30)

        # calculate episodic reduction in epsilon
        self.epsilon_step = (epsilon - min_eps) / num_episodes

        # reward logging
        self.reward_log_columns = [
            'Episode',
            'Ep start ts',
            'Feedback ts',
            'Human Reward',
            'Environment Reward',
        ]
        self.reward_log_path = os.path.join(self.output_dir, f'{self.uuid}.csv')

    def __del__(self):
        print('Closing the agent')
        self.env.stop_reading_feedback()

    def act(self, state):
        """ Epsilon-greedy Policy """
        if np.random.random() < 1 - self.epsilon:
            preds = self.H.predict(state) if self.interactive else self.Q.predict(state)
            return int(np.argmax(preds))
        else:
            return np.random.randint(0, self.env.action_space.n)

    def _train_episode(self, episode_index):
        print(f'Episode: {episode_index + 1}  Timestep:', end='')
        rng = np.random.default_rng()
        tot_reward = 0
        state = self.env.reset()[0]
        ep_start_time = dt.datetime.now().time()
        with open(self.reward_log_path, 'a+', newline='') as write_obj:
            dict_writer = DictWriter(write_obj, fieldnames=self.reward_log_columns)
            dict_writer.writeheader()
            for ts in count():
                print(f' {ts}', end='')
                self.env.show_simulation()

                # Determine next action
                action = self.act(state)

                # Save the state and state occurrence time
                self.transitions_buffer.append((state, time.time(), action))

                if self.interactive:
                    self.env.show_action(action)

                # Get next state and reward
                next_state, reward, done, truncated, info = self.env.step(action)

                if not self.interactive:
                    if done and next_state[0] >= 0.5:
                        td_target = reward
                    else:
                        td_target = reward + self.discount_factor * np.max(
                            self.Q.predict(next_state)
                        )
                    self.Q.update(state, action, td_target)
                else:
                    now = time.time()
                    update_flag = True
                    while time.time() < now + self.ts_len:
                        time.sleep(0.01)  # save the CPU

                        human_reward = self.env.get_scalar_feedback()
                        human_reward_time = time.time()

                        if self.interactive:
                            self.env.show_action(action)

                        feedback_ts = dt.datetime.now().time()
                        if human_reward != 0 and update_flag:
                            dict_writer.writerow(
                                {
                                    'Episode': episode_index + 1,
                                    'Ep start ts': ep_start_time,
                                    'Feedback ts': feedback_ts,
                                    'Human Reward': human_reward,
                                    'Environment Reward': reward
                                }
                            )

                            self.H.update(state=state,
                                          next_state=next_state,
                                          action=action,
                                          td_target=human_reward,
                                          transitions_time=self.transitions_buffer,
                                          feedback_time=human_reward_time)
                            update_flag = False

                tot_reward += reward
                if done:
                    print(f'  Reward: {tot_reward}')
                    break

                stdout.write('\b' * (len(str(ts)) + 1))
                state = next_state

        # Decay epsilon
        if self.epsilon > self.min_eps:
            self.epsilon -= self.epsilon_step

    def train(self, model_file_to_save=None):
        """
        H (or Q learning) training loop
        Args:
            model_file_to_save: save Q or H model to this filename
        """
        # render first so that pygame display shows up on top
        self.env.show_simulation()

        for i in range(self.num_episodes):
            self._train_episode(i)

        print('\nCleaning up...')
        self.env.stop_reading_feedback()
        if model_file_to_save is not None:
            self.save_model(filename=model_file_to_save)

    def play(self, n_episodes=1, render=True):
        """
        Run episodes with trained agent
        Args:
            n_episodes: number of episodes
            render: optionally render episodes

        Returns: list of cumulative episode rewards
        """
        self.epsilon = 0
        ep_rewards = []
        for i in range(n_episodes):
            state = self.env.reset()[0]
            done = False
            tot_reward = 0
            while not done:
                action = self.act(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                tot_reward += reward
                if render:
                    self.env.show_simulation()
                    time.sleep(0.1)
                state = next_state
            ep_rewards.append(tot_reward)
            print(f'Episode: {i + 1} Reward: {tot_reward}')
        #self.env.close()
        return ep_rewards

    def evaluate(self, n_episodes=100):
        print('Evaluating agent')
        rewards = self.play(n_episodes=n_episodes)
        avg_reward = np.mean(rewards)
        print(
            f'Average total episode reward over {n_episodes} '
            f'episodes: {avg_reward:.2f}'
        )
        return avg_reward

    def save_model(self, filename):
        """
        Save H or Q model to models dir
        Args:
            filename: name of pickled file
        """
        model = self.H if self.interactive else self.Q
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(self.model_file_to_save.joinpath(filename), 'wb') as f:
            pickle.dump(model, f)
        self.model_file_to_load = self.model_file_to_save

    def load_model(self, filename):
        """
        Load H or Q model from models dir
        Args:
            filename: name of pickled file
        """
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(self.model_file_to_load.joinpath(filename), 'rb') as f:
            model = pickle.load(f)
        if self.interactive:
            self.H = model
        else:
            self.Q = model
