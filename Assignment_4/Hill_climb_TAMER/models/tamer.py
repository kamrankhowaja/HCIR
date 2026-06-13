"""
    Copyright 2024 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of pyirl.
    It contains implementation of the TAMER algorithm.

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
import numpy as np
from models.utils.credit_assignment import CreditAssignment


class Tamer(InteractiveModel):
    def __init__(self, env, human_answer_interval=None, credit_assignment=False):
        self.model = SGDFunctionApproximator(env)
        self.if_ca = credit_assignment
        if self.if_ca:
            self.ca = CreditAssignment(min_delay=human_answer_interval[0],
                                       max_delay=human_answer_interval[1])

    def predict(self, state, action=None):
        return self.model.predict(state=state, action=None)

    def update(self, action, td_target, state=None, next_state=None, transitions_time=None, feedback_time=None):
        if self.if_ca:
            if transitions_time is None or feedback_time is None:
                raise ValueError("Credit assignment activated. "
                                 "Must provide transitions_time and feedback_time arguments")

            states, credits, _ = self.ca.calculate_credits(transitions_time=transitions_time,
                                                           feedback_time=feedback_time)
            f_states = np.array([self.model.featurize_state(state) for state in states])
            update_f_state = np.sum(f_states * credits, axis=0)
            self.model.update(state=update_f_state, action=action, td_target=td_target, if_featurize=False)
        else:
            if state is None:
                raise ValueError("Credit assignment deactivated. "
                                 "Must provide state arguments")
            self.model.update(state=state, action=action, td_target=td_target, if_featurize=True)
