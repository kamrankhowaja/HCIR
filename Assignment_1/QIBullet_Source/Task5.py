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
        # Arm movement to wave hand
        self.nao.setAngles("RShoulderPitch", -1.0, 0.2)
        # self.nao.setAngles("RShoulderRoll", -0.3, 0.2)
        self.nao.setAngles("RElbowYaw", 1.0, 0.2)
        # self.nao.setAngles("RElbowRoll", 1.2, 0.2)
        time.sleep(0.5)

        for i in range(3):
            self.nao.setAngles("RWristYaw", 1.0, 0.3)
            time.sleep(0.5)
            self.nao.setAngles("RWristYaw", -1.0, 0.3)
            time.sleep(0.5)

        # Return arm to initial position
        self.nao.setAngles("RShoulderPitch", 1.0, 0.2)
        # self.nao.setAngles("RShoulderRoll", 0.3, 0.2)
        self.nao.setAngles("RElbowYaw", -1.0, 0.2)
        # self.nao.setAngles("RElbowRoll", -1.2, 0.2)
        time.sleep(0.5)

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