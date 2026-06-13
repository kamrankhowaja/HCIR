"""
    Copyright 2024 by Michał Stolarz <michal.stolarz@h-brs.de>

    This file is part of pyirl,
    and is based on: https://github.com/benibienz/TAMER.
    It contains the environment class for the interactive agent.

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

import os
import pygame
import numpy as np
from threading import Thread, Event, Lock


class Timer(Thread):
    def __init__(self, t, function, event):
        Thread.__init__(self)
        self.stopped = event
        self.function = function
        self.t = t

    def run(self):
        while not self.stopped.wait(self.t):
            self.function()


class Environment:
    """ Environment for training the agent """

    def __init__(self, environment, action_map, human_feedback_read_t=0.1):
        self.env = environment
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.action_map = action_map
        pygame.init()
        self.font = pygame.font.Font("freesansbold.ttf", 60)

        # set position of pygame window (so it doesn't overlap with gym)
        os.environ["SDL_VIDEO_WINDOW_POS"] = "1000,100"
        os.environ["SDL_VIDEO_CENTERED"] = "0"

        self.screen = pygame.display.set_mode((600, 600))
        self.reward = 0
        pygame.display.set_caption('Control Interface')
        area = self.screen.fill((0, 0, 0))
        self.action = 0
        pygame.display.update(area)
        self.lock = Lock()
        self.stop_reading_feedback_event = Event()
        self.timed_feedback = Timer(t=human_feedback_read_t,
                                    function=self.read_feedback,
                                    event=self.stop_reading_feedback_event)
        self.timed_feedback.start()

    def __del__(self):
        self.stop_reading_feedback()
        pygame.quit()

    def stop_reading_feedback(self):
        self.stop_reading_feedback_event.set()
        self.lock.acquire()
        self.screen.fill((128, 128, 128), (0, 0, self.screen.get_width(), 200))
        text = self.font.render("Evaluating", True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = (290, 90)
        area = self.screen.blit(text, text_rect)
        pygame.display.update(area)
        self.lock.release()

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def get_scalar_feedback(self):
        self.lock.acquire()
        reward = self.reward
        self.lock.release()
        return reward

    def read_feedback(self):
        """
        Get human input. 'W' key for positive, 'A' key for negative.
        Returns: scalar reward (1 for positive, -1 for negative)
        """
        reward = 0
        self.lock.acquire()
        area = self.screen.fill((128, 128, 128), (0, 0, self.screen.get_width(), 200))
        #TODO: Fill in the for loop below and implement human feedback as per the docstring
        # for event in pygame.event.get():
        #     if event.type == pygame.KEYDOWN:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stop_reading_feedback()
                pygame.quit()
                raise SystemExit

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    reward = 1
                    print("Human feedback: +1")

                elif event.key == pygame.K_a:
                    reward = -1
                    print("Human feedback: -1")

        pygame.display.update(area)
        self.reward = reward
        self.lock.release()

    def show_action(self, action):
        """
        Show agent's action on pygame screen
        Args:
            action: numerical action (for MountainCar environment only currently)
        """
        self.lock.acquire()
        text = self.font.render(self.action_map[action], True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = (290, 90)
        area = self.screen.blit(text, text_rect)
        pygame.display.update(area)
        self.lock.release()

    def show_simulation(self):
        self.lock.acquire()
        image = self.env.render()
        image = np.moveaxis(image, 0, 1)
        surf = pygame.surfarray.make_surface(image)
        area = self.screen.blit(surf, (0, 200))
        pygame.display.update(area)
        self.lock.release()
