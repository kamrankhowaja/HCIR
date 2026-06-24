"""
Robot Interaction Layer for EventMate.

Provides:
  - TTS  : robot_say()
  - STT  : robot_listen()
  - Gestures : robot_wave(), robot_nod(), robot_thinking(),
               robot_thumbs_up(), robot_excited(), robot_farewell()
  - Simulation launcher : launch_simulation()

"""

from __future__ import annotations
import base64
import shutil
import subprocess
import sys
import time
import threading
from pathlib import Path

# TTS
_tts_engine = None
_POWERSHELL = shutil.which("powershell.exe")
_IS_WSL = "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()

_POWERSHELL_AVAILABLE = False
if _IS_WSL and _POWERSHELL:
    try:
        _POWERSHELL_AVAILABLE = subprocess.run(
            [_POWERSHELL, "-NoProfile", "-Command", "exit 0"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).returncode == 0
    except Exception:
        _POWERSHELL_AVAILABLE = False

if _IS_WSL and _POWERSHELL_AVAILABLE:
    TTS_AVAILABLE = True
    TTS_BACKEND = "windows-sapi"
else:
    try:
        import pyttsx3
        _tts_engine = pyttsx3.init()
        _tts_engine.setProperty("rate", 150)
        _tts_engine.setProperty("volume", 1.0)
        TTS_AVAILABLE = True
        TTS_BACKEND = "pyttsx3"

        if sys.platform.startswith("linux"):
            if shutil.which("aplay") is None:
                TTS_AVAILABLE = False
                TTS_BACKEND = None
                print("[WARN] TTS playback unavailable: 'aplay' was not found.")
                print("[WARN] Install ALSA playback tools with: sudo apt install alsa-utils")
            elif _IS_WSL:
                alsa_devices = subprocess.run(
                    ["aplay", "-L"],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout
                if "pulse" not in alsa_devices and not Path("/dev/snd").exists():
                    TTS_AVAILABLE = False
                    TTS_BACKEND = None
                    print("[WARN] TTS playback unavailable: WSL has no ALSA output device.")
                    print("[WARN] Install the Pulse ALSA bridge with: sudo apt install libasound2-plugins pulseaudio-utils")
    except Exception as e:
        TTS_AVAILABLE = False
        TTS_BACKEND = None
        print(f"[WARN] TTS unavailable: {e}")
        print("[WARN] Install pyttsx3 plus a system speech backend such as eSpeak/eSpeak-ng.")

# STT (optional)
try:
    import speech_recognition as sr
    _recognizer = sr.Recognizer()
    _microphone = sr.Microphone()
    STT_AVAILABLE = True
except Exception:
    STT_AVAILABLE = False
    print("[INFO] SpeechRecognition/pyaudio not available — keyboard fallback active.")

# qiBullet
try:
    from qibullet import SimulationManager
    QIBULLET_AVAILABLE = True
except Exception:
    QIBULLET_AVAILABLE = False
    print("[WARN] qiBullet not available — gestures will be printed only.")


#  SIMULATION LAUNCHER
def launch_simulation():
    """
    Start the qiBullet simulation and return (pepper, sim_manager).
    Returns (None, None) if qiBullet is unavailable.
    """
    if not QIBULLET_AVAILABLE:
        print("[INFO] qiBullet not available — running without simulation.")
        return None, None

    sim_manager = SimulationManager()
    client_id = sim_manager.launchSimulation(gui=True)
    pepper = sim_manager.spawnPepper(client_id, spawn_ground_plane=True)
    pepper.goToPosture("StandInit", 0.6)
    time.sleep(1.5)
    print("[INFO] Pepper spawned in qiBullet.")
    return pepper, sim_manager

#  SPEECH: TTS + STT - WSL related stuff
def _windows_sapi_say(text: str) -> None:
    """Speak through the Windows host when running inside WSL."""
    encoded_text = base64.b64encode(text.encode("utf-16le")).decode("ascii")
    script = f"""
            Add-Type -AssemblyName System.Speech
            $text = [System.Text.Encoding]::Unicode.GetString([Convert]::FromBase64String('{encoded_text}'))
            $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
            $synth.Rate = 0
            $synth.Volume = 100
            $synth.Speak($text)
        """
    encoded_script = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    subprocess.run(
        [_POWERSHELL, "-NoProfile", "-EncodedCommand", encoded_script],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def robot_say(text: str, pepper=None) -> None:
    """Print + speak text via TTS."""
    print(f"[PEPPER] {text}")
    if TTS_AVAILABLE:
        try:
            if TTS_BACKEND == "windows-sapi":
                _windows_sapi_say(text)
            elif TTS_BACKEND == "pyttsx3":
                _tts_engine.say(text)
                _tts_engine.runAndWait()
        except Exception as e:
            print(f"[WARN] TTS error: {e}")


def robot_act_and_say(
    action_fn,
    text: str,
    pepper=None,
    *,
    action_delay: float = 0.0,
    speech_delay: float = 0.25,
) -> None:
    """
    Start a gesture and speech with a small offset for natural timing.
    """
    def _delayed_action():
        if action_delay > 0:
            time.sleep(action_delay)
        try:
            action_fn(pepper)
        except Exception as e:
            print(f"[WARN] Action error: {e}")

    threading.Thread(target=_delayed_action, daemon=True).start()

    if speech_delay > 0:
        time.sleep(speech_delay)
    robot_say(text, pepper)


def robot_listen(prompt_text: str, pepper=None) -> str:
    """
    Speak prompt then listen for user response.
    Tries microphone first, falls back to keyboard.
    """
    robot_say(prompt_text, pepper)

    if STT_AVAILABLE:
        robot_say("I am listening...", pepper)
        try:
            with _microphone as source:
                _recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("[INFO] Listening via microphone...")
                audio = _recognizer.listen(source, timeout=8, phrase_time_limit=6)
            text = _recognizer.recognize_google(audio)
            print(f"[USER via mic] {text}")
            return text.strip()
        except Exception as e:
            print(f"[INFO] Mic failed ({e}) — switching to keyboard.")

    answer = input("[YOU] ").strip()
    return answer

#  GESTURES  — all run in daemon threads so they don't block the main loop
def _run_in_thread(fn):
    """Helper: run fn in a daemon thread."""
    threading.Thread(target=fn, daemon=True).start()


# WAVE
def robot_wave(pepper=None) -> None:
    """Right-arm greeting wave."""
    if not QIBULLET_AVAILABLE or pepper is None:
        print("[PEPPER] *waves*")
        return

    def _wave():
        try:
            # Raise arm
            pepper.setAngles(
                ["RShoulderPitch", "RShoulderRoll", "RElbowRoll"],
                [-0.5, -0.3, 1.2], 0.4
            )
            time.sleep(1.0)
            # Wave side-to-side
            for _ in range(3):
                pepper.setAngles(["RShoulderRoll"], [-0.8], 0.6)
                time.sleep(0.35)
                pepper.setAngles(["RShoulderRoll"], [-0.1], 0.6)
                time.sleep(0.35)
            # Return to rest
            pepper.setAngles(
                ["RShoulderPitch", "RShoulderRoll", "RElbowRoll"],
                [1.5, -0.1, 0.5], 0.3
            )
        except Exception as e:
            print(f"[WARN] Wave error: {e}")

    _run_in_thread(_wave)


# NOD
def robot_nod(pepper=None, times: int = 2) -> None:
    """Head nod — acknowledgement."""
    if not QIBULLET_AVAILABLE or pepper is None:
        print("[PEPPER] *nods*")
        return

    def _nod():
        try:
            for _ in range(times):
                pepper.setAngles(["HeadPitch"], [0.3], 0.4)
                time.sleep(0.45)
                pepper.setAngles(["HeadPitch"], [-0.05], 0.4)
                time.sleep(0.45)
            pepper.setAngles(["HeadPitch"], [0.0], 0.3)
        except Exception as e:
            print(f"[WARN] Nod error: {e}")

    _run_in_thread(_nod)


# THINKING
def robot_thinking(pepper=None, duration: float = 2.5) -> None:
    """
    Hand-to-chin thinking pose — used while BN inference runs.
    Blocks for `duration` seconds (intentional: inference is happening).
    """
    if not QIBULLET_AVAILABLE or pepper is None:
        print("[PEPPER] *thinking...*")
        time.sleep(duration)
        return

    try:
        # Look slightly downward
        pepper.setAngles(["HeadPitch"], [0.25], 0.3)
        # Raise right hand toward chin
        pepper.setAngles(
            ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw"],
            [-0.2, -0.15, 1.3, 1.0], 0.3
        )
        time.sleep(duration)
        # Reset
        pepper.setAngles(["HeadPitch"], [0.0], 0.3)
        pepper.setAngles(
            ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw"],
            [1.5, -0.1, 0.5, 0.0], 0.3
        )
    except Exception as e:
        print(f"[WARN] Thinking error: {e}")


# THUMBS UP
def robot_thumbs_up(pepper=None) -> None:
    """Enthusiastic thumbs-up — used when showing top recommendation."""
    if not QIBULLET_AVAILABLE or pepper is None:
        print("[PEPPER] *thumbs up!*")
        return

    def _thumbs_up():
        try:
            pepper.setAngles(
                ["RShoulderPitch", "RShoulderRoll", "RElbowRoll"],
                [0.2, -0.15, 1.5], 0.35
            )
            time.sleep(1.5)
            pepper.setAngles(
                ["RShoulderPitch", "RShoulderRoll", "RElbowRoll"],
                [1.5, -0.1, 0.5], 0.3
            )
        except Exception as e:
            print(f"[WARN] Thumbs-up error: {e}")

    _run_in_thread(_thumbs_up)


# EXCITED
def robot_excited(pepper=None) -> None:
    """Both-arms raise — used when greeting a known returning user."""
    if not QIBULLET_AVAILABLE or pepper is None:
        print("[PEPPER] *excited!*")
        return

    def _excited():
        try:
            for _ in range(2):
                pepper.setAngles(
                    ["RShoulderPitch", "LShoulderPitch"],
                    [-0.5, -0.5], 0.3
                )
                time.sleep(0.4)
                pepper.setAngles(
                    ["RShoulderPitch", "LShoulderPitch"],
                    [0.3, 0.3], 0.3
                )
                time.sleep(0.4)
            # Reset
            pepper.setAngles(
                ["RShoulderPitch", "LShoulderPitch"],
                [1.5, 1.5], 0.3
            )
        except Exception as e:
            print(f"[WARN] Excited error: {e}")

    _run_in_thread(_excited)


# FAREWELL WAVE
def robot_farewell(pepper=None) -> None:
    """
    Slow, deliberate farewell wave — bigger motion than greeting wave.
    Blocks briefly so farewell speech overlaps naturally.
    """
    if not QIBULLET_AVAILABLE or pepper is None:
        print("[PEPPER] *farewell wave*")
        return

    def _farewell():
        try:
            # Raise arm high
            pepper.setAngles(
                ["RShoulderPitch", "RShoulderRoll", "RElbowRoll"],
                [-0.8, -0.5, 0.8], 0.35
            )
            time.sleep(0.8)
            # Slow wave 3×
            for _ in range(3):
                pepper.setAngles(["RShoulderRoll"], [-0.9], 0.4)
                time.sleep(0.5)
                pepper.setAngles(["RShoulderRoll"], [-0.2], 0.4)
                time.sleep(0.5)
            # Return to stand
            pepper.goToPosture("StandInit", 0.4)
        except Exception as e:
            print(f"[WARN] Farewell error: {e}")

    _run_in_thread(_farewell)


# POINT FORWARD
def robot_point(pepper=None) -> None:
    """Point forward — used when presenting a recommendation."""
    if not QIBULLET_AVAILABLE or pepper is None:
        print("[PEPPER] *points forward*")
        return

    def _point():
        try:
            pepper.setAngles(
                ["RShoulderPitch", "RElbowRoll", "RElbowYaw"],
                [-0.8, 0.3, 0.0], 0.35
            )
            time.sleep(1.5)
            pepper.setAngles(
                ["RShoulderPitch", "RElbowRoll"],
                [1.5, 0.5], 0.3
            )
        except Exception as e:
            print(f"[WARN] Point error: {e}")

    _run_in_thread(_point)
