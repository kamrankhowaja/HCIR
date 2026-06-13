"""
    Copyright 2024 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of pyirl.
    It contains an abstract class that defines an interface for an interactive model.

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

from abc import ABC, abstractmethod


class InteractiveModel(ABC):
    @abstractmethod
    def predict(self, state, action):
        pass

    @abstractmethod
    def update(self, state, action, feedback, next_state, states_time, feedback_time):
        pass

