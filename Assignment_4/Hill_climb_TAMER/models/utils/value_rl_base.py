"""
    Copyright 2024 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of pyirl.
    It contains implementation of the value-based reinforcement learning algorithm.

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

from models.utils.interactive_model import InteractiveModel
from models.utils.sgd_function_approximator import SGDFunctionApproximator
from abc import abstractmethod


class ValueBasedRL(InteractiveModel):
    def __init__(self, env):
        self.model = SGDFunctionApproximator(env)
        self.discount_factor = 0.8

    def predict(self, state, action=None):
        return self.model.predict(action=action, state=state)

    @abstractmethod
    def update(self, state, action, td_target, next_state=None, states_time=None, feedback_time=None):
        pass
