import os
import time
import pygame
import threading
from gtts import gTTS
from qibullet import SimulationManager, PepperVirtual

def execute_after_delay(delay, action, *args):
    """Utility to trigger a behavior at a specific BML start time."""
    time.sleep(delay)
    action(*args)

class MultiModelPepper:
    def __init__(self):
        self.simulation_manager = SimulationManager()
        self.client = self.simulation_manager.launchSimulation(gui=True)
        self.pepper = self.simulation_manager.spawnPepper(self.client, spawn_ground_plane=True)

        # BML dictionary
        self.behavior_bml = {
            "gaze": {
                "start": 0.0,
                "duration": 4.0,
                "pitch": -0.2
            },
            "wave": {
                "start": 0.5,
                "duration": 2.5
            },
            "speech": [
                {
                    "start": 0.7,
                    "text": "Hello!"
                },
                {
                    "start": 3.0,
                    "text": "Glad to see you!"
                }
            ],
            "nod": {
                "start": 3.0,
                "duration": 2.0
            },
            "swirl": {
                "start": 5.5,
                "duration": 2.0
            }
        }
    
    def speak(self, text):
        '''Converts text to speech and plays it'''
        filename = f"pepper_audio_{threading.get_ident()}.mp3"

        # generate speech audio
        tts = gTTS(text=text, lang="en")
        with open(filename, "wb") as f:
            tts.write_to_fp(f)

        # plays audio file
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        # Wait until speech finishes
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.quit()

        # Delete temporary file
        os.remove(filename)

    def gaze_at_human(self):
        '''Pepper slightly lifts head upwards'''
        self.pepper.setAngles("HeadPitch", -0.2, 0.15)
        time.sleep(4)
    
    def wave_right_hand(self):
        '''Pepper performs right-hand waving gesture'''

        # Raise arm
        self.pepper.setAngles("RShoulderPitch", 0.0, 0.15)
        self.pepper.setAngles("RShoulderRoll", -1.3, 0.15)
        self.pepper.setAngles("RElbowYaw", 1.5, 0.15)
        self.pepper.setAngles("RElbowRoll", 1.5, 0.15)
        self.pepper.setAngles("RWristYaw", 0.0, 0.15)
        self.pepper.setAngles("RHand", 1.0, 0.15)

        time.sleep(1.2)

        # Wave motion
        for _ in range(4):
            self.pepper.setAngles("RElbowRoll", 0.9, 0.4)
            time.sleep(0.4)
            self.pepper.setAngles("RElbowRoll",1.5,0.4)
            time.sleep(0.4)

        # Return arm to normal
        self.reset_right_arm()
    
    def nod_head(self):
        '''Pepper performs head nodding motion'''
        for _ in range(3):
            # Head down
            self.pepper.setAngles("HeadPitch", 0.3, 0.2)
            time.sleep(0.4)

            # Head up
            self.pepper.setAngles("HeadPitch",-0.1,0.2)
            time.sleep(0.4)

    def happy_swirl(self):
        '''Pepper performs happy body swirl motion'''
        # Rotate left
        self.pepper.setAngles("HipRoll", 0.25,0.15)
        time.sleep(0.6)

        # Rotate right
        self.pepper.setAngles("HipRoll", -0.25,0.15)
        time.sleep(0.6)

        # Return center
        self.pepper.setAngles("HipRoll",0.0,0.15)

    def reset_right_arm(self):
        '''Resets Pepper's arm to default pose'''

        self.pepper.setAngles("RHand", 0.0, 0.15)
        self.pepper.setAngles("RShoulderPitch", 1.5, 0.15)
        self.pepper.setAngles("RShoulderRoll", -0.1, 0.15)
        self.pepper.setAngles("RElbowYaw", 1.2, 0.15)
        self.pepper.setAngles("RElbowRoll", 0.5, 0.15)

    def reset_posture(self):
        '''Returns Pepper to default posture'''

        self.pepper.setAngles("HeadPitch", 0.0, 0.15)
        self.pepper.setAngles("HipRoll", 0.0, 0.15)

        self.reset_right_arm()

    def execute_behavior(self):
            threads = []

            # GAZE
            threads.append(
                threading.Thread(
                    target=execute_after_delay,
                    args=(
                        self.behavior_bml["gaze"]["start"],
                        self.gaze_at_human
                    )
                )
            )

            # WAVE
            threads.append(
                threading.Thread(
                    target=execute_after_delay,
                    args=(
                        self.behavior_bml["wave"]["start"],
                        self.wave_right_hand
                    )
                )
            )

            # SPEECH 1
            threads.append(
                threading.Thread(
                    target=execute_after_delay,
                    args=(
                        self.behavior_bml["speech"][0]["start"],
                        self.speak,
                        self.behavior_bml["speech"][0]["text"]
                    )
                )
            )

            # SPEECH 2
            threads.append(
                threading.Thread(
                    target=execute_after_delay,
                    args=(
                        self.behavior_bml["speech"][1]["start"],
                        self.speak,
                        self.behavior_bml["speech"][1]["text"]
                    )
                )
            )

            # NOD
            threads.append(
                threading.Thread(
                    target=execute_after_delay,
                    args=(
                        self.behavior_bml["nod"]["start"],
                        self.nod_head
                    )
                )
            )

            # SWIRL
            threads.append(
                threading.Thread(
                    target=execute_after_delay,
                    args=(
                        self.behavior_bml["swirl"]["start"],
                        self.happy_swirl
                    )
                )
            )

            # Start all behaviors
            for thread in threads:
                thread.start()

            # Wait for all behaviors to finish
            for thread in threads:
                thread.join()

            # Reset Pepper posture
            self.reset_posture()

# main execution
if __name__ == "__main__":
    pepper_behavior = MultiModelPepper()
    print("Starting multimodal behavior...")

    pepper_behavior.execute_behavior()
    print("Behavior execution completed.")