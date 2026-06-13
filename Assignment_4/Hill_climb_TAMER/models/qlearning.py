"""
    Copyright 2024 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of pyirl.
    It contains implementation of the Qlearning algorithm.

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

from models.utils.value_rl_base import ValueBasedRL
import numpy as np


class Qlearning(ValueBasedRL):
    """
    Qlearning - without exploration the update step is same as SARSA
    """
    def __init__(self, env):
        ValueBasedRL.__init__(self, env)

    def update(self, state, action, td_target, next_state=None, transitions_time=None, feedback_time=None):
        q_pred = np.max(self.predict(next_state))
        q = self.predict(state=state, action=action)
        td_target = td_target + self.discount_factor * q_pred - q
        self.model.update(state=state, action=action, td_target=td_target, if_featurize=True)
