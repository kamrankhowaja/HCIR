from qibullet import SimulationManager
from gtts import gTTS
# from playsound import playsound
import pygame
import time
import threading
import os

class BehaviorRealizer():

    def __init__(self):
        simulation_manager = SimulationManager()
        client = simulation_manager.launchSimulation(gui=True)
        self.nao = simulation_manager.spawnNao(
            client,
            spawn_ground_plane=True
        )

    def speak(self, text):
        filename = f"robot_speech.mp3"

        tts = gTTS(text=text, lang="en")
        tts.save(filename)

        # Here i would like to mention that i use Pygame as Playsound was not working for me 
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.quit()
        os.remove(filename)


    def waving_hand(self):
        # Raise arm into L-shape
        self.nao.setAngles("RShoulderPitch", 0.0,  0.15)
        self.nao.setAngles("RShoulderRoll",  -1.3, 0.15)
        self.nao.setAngles("RElbowYaw",       1.5, 0.15)
        self.nao.setAngles("RElbowRoll",      1.5, 0.15)
        self.nao.setAngles("RWristYaw",       0.0, 0.15)
        self.nao.setAngles("RHand",           1.0, 0.15) 

        # wait for arm to reach
        time.sleep(1.2)

        # Wave
        for _ in range(4):
            self.nao.setAngles("RElbowRoll", 0.9, 0.4)   # open elbow
            time.sleep(0.45)
            self.nao.setAngles("RElbowRoll", 1.5, 0.4)   # close elbow
            time.sleep(0.45)

        # Return arm
        self.nao.setAngles("RHand",           0.0, 0.15)
        self.nao.setAngles("RShoulderPitch", 1.5,  0.15)
        self.nao.setAngles("RShoulderRoll",  -0.1, 0.15)
        self.nao.setAngles("RElbowYaw",       1.2, 0.15)
        self.nao.setAngles("RElbowRoll",      0.5, 0.15)
        self.nao.setAngles("RWristYaw",       0.0, 0.15)
        time.sleep(1.0)

if __name__ == "__main__":

    behavior_realizer_class = BehaviorRealizer()

    INPUT_OPTIONS = ["done",'exit','end','stop']

    repeat_ = True
    while repeat_:
        user_input = input("INPUT: (Write done exit end or stop to end the program) ")

        if user_input.lower() in INPUT_OPTIONS:
            repeat_ = False

        else:
            behavior_realizer_class.speak(user_input)
            behavior_realizer_class.waving_hand()