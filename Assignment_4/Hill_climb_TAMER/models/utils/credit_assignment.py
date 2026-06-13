"""
    Copyright 2024 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of pyirl.
    It contains implementation of credit assignment technique.

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

import numpy as np
from models.utils.utils import TransitionsFilter


class CreditAssignment:
    """
    Credit assignment.
    """
    def __init__(self, min_delay, max_delay):
        self.filter = TransitionsFilter(min_delay, max_delay)

    def calculate_credits(self, transitions_time, feedback_time):
        filtered_transitions = self.filter.filter(transitions_time, feedback_time)
        states = np.array([transition[0] for transition in filtered_transitions])
        actions = np.array([transition[1] for transition in filtered_transitions])
        credit_vals = np.array([[1/len(states)]]*len(states)) #assumption of the uniform distribution
        return states, credit_vals, actions
