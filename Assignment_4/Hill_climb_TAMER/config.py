"""
    Copyright 2024 by Michał Stolarz <michal.stolarz@h-brs.de>, Ritwik Sinha <ritwik.sinha@smail.inf.h-brs.de>

    This file is part of pyirl
    and is based on: https://github.com/benibienz/TAMER.
    It contains the configuration file.

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


# If irl model should be used, otherwise use vanilla Qlearning
IF_INTERACTIVE = True

#Available models: tamer, qlearning

#TODO: enter the model you wish to train
IRL_MODEL = "tamer"

# Credit assignment status: True, False

#TODO: set the credit assignment status
CA_STATUS = True  #False for task 2, True for Task 1 

# set a timestep for training the agent
# the more time per step, the easier for the human
# but the longer it takes to train (in real time)
# 0.2 seconds is fast but doable
TRAINING_TIMESTEP = 0.15

# Human response time, based on Knox + Stone 2009
HUMAN_ANSWER_INTERVAL = (0.2, 0.6)

# frequency of reading the human feedback
HUMAN_FEEDBACK_READ_TIME = 0.1
