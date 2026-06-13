"""
    Copyright 2024 by Michał Stolarz <michal.stolarz@h-brs.de>, Ritwik Sinha <ritwik.sinha@smail.inf.h-brs.de>

    This file is part of pyirl,
    and is based on: https://github.com/benibienz/TAMER.
    It contains the script to run interactive machine learning algorithms.

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


"""
When training, use 'W' and 'A' keys for positive and negative rewards
"""

import gymnasium as gym
from agent.agent import Agent
from environment.environment import Environment
from pathlib import Path
import config as cfg

MODELS_DIR = Path(__file__).parent.joinpath('saved_models')
LOGS_DIR = Path(__file__).parent.joinpath('logs')

def main():
    gym_env = gym.make('MountainCar-v0', render_mode='rgb_array')
    gym_env.reset()
    mountaincar_action_map = {0: 'left', 1: 'none', 2: 'right'}
    environment = Environment(environment=gym_env,
                              action_map=mountaincar_action_map,
                              human_feedback_read_t=cfg.HUMAN_FEEDBACK_READ_TIME)

    # hyperparameters
    discount_factor = 1
    epsilon = 0  # vanilla Q learning actually works well with no random exploration
    min_eps = 0
    num_episodes = 10 # change to 10 or 20  for task 2 
    interactive = cfg.IF_INTERACTIVE
    model = cfg.IRL_MODEL
    credit_assignment = cfg.CA_STATUS

    training_timestep = cfg.TRAINING_TIMESTEP

    agent = Agent(env=environment,
                  model=model,
                  num_episodes=num_episodes,
                  credit_assignment = credit_assignment,
                  discount_factor=discount_factor,
                  epsilon=epsilon,
                  min_eps=min_eps,
                  interactive=interactive,
                  ts_len=training_timestep,
                  human_answer_interval=cfg.HUMAN_ANSWER_INTERVAL,
                  output_dir=LOGS_DIR,
                  model_file_to_save=MODELS_DIR,
                  model_file_to_load=None)

    agent.train(model_file_to_save='autosave')
    agent.play(n_episodes=1, render=True)
    agent.evaluate(n_episodes=30)


if __name__ == '__main__':
    main()




