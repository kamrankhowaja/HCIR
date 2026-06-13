"""
    Copyright 2024 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of pyirl.
    It contains utils additional functionalities for pyirl.

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


class TransitionsFilter:
    """
    Class for filtering transitions with respect to the human feedback delay constraints.
    """
    def __init__(self,
                 min_delay=0.2,
                 max_delay=0.6):
        self.min_delay = min_delay
        self.max_delay = max_delay

    def filter(self, transitions_time, feedback_time):
        filtered_transitions = []
        for state, state_time, action in transitions_time:
            delay = feedback_time - state_time
            if self.min_delay < delay < self.max_delay:
                filtered_transitions.append((state, action))

        return filtered_transitions
