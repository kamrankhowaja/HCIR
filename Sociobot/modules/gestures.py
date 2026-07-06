"""
Robot Interaction Layer for EventMate
"""

from __future__ import annotations
import asyncio
import base64
import io
import shutil
import subprocess
import sys
import time
import random
import threading
from pathlib import Path
import numpy as np

# Edge TTS
EDGE_TTS_VOICE = "en-GB-SoniaNeural"
try:
    import edge_tts as _edge_tts_module
    import soundfile as sf
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    print("[WARN] edge-tts not available. Install with: pip install edge-tts soundfile")

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

# STT (optional) - microphone capture via sounddevice, transcription via faster-whisper
try:
    import sounddevice as sd
    sd.query_devices(kind="input")
    STT_AVAILABLE = True
except Exception as e:
    STT_AVAILABLE = False
    print(f"[INFO] Microphone capture unavailable ({e}) - keyboard fallback active.")

try:
    from faster_whisper import WhisperModel
    _whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
    WHISPER_AVAILABLE = True
except Exception as e:
    WHISPER_AVAILABLE = False
    _whisper_model = None
    print(f"[WARN] faster-whisper unavailable: {e}")
    print("[WARN] Install with: pip install faster-whisper")

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

    return pepper, (sim_manager, client_id)

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

async def _edge_tts_speak_async(text: str, voice: str) -> None:
    """Stream audio from edge-tts and play via sounddevice."""
    communicate = _edge_tts_module.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    data, samplerate = sf.read(buf)
    sd.play(data, samplerate)
    sd.wait()

def robot_say(text: str, pepper=None) -> None:
    """Print + speak text. Priority: edge-tts > windows-sapi > pyttsx3"""
    print(f"[SOCIOBOT] {text}")

    # edge-tts (needs internet)
    if EDGE_TTS_AVAILABLE:
        try:
            asyncio.run(_edge_tts_speak_async(text, EDGE_TTS_VOICE))
            return
        except Exception as e:
            print(f"[WARN] edge-tts error: {e} — falling back to system TTS.")

    # Windows SAPI (WSL only)
    if TTS_BACKEND == "windows-sapi":
        try:
            _windows_sapi_say(text)
            return
        except Exception as e:
            print(f"[WARN] Windows SAPI error: {e}")

    # pyttsx3 (last resort)
    if TTS_BACKEND == "pyttsx3" and _tts_engine is not None:
        try:
            _tts_engine.say(text)
            _tts_engine.runAndWait()
        except Exception as e:
            print(f"[WARN] pyttsx3 error: {e}")


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

def robot_listening_gesture(pepper=None):
    """
    Non-verbal listening cue while the user is speaking.
    Returns a stop function.
    """
    stop_event = threading.Event()

    if not QIBULLET_AVAILABLE or pepper is None:
        print("[SOCIOBOT] *listening with a small nod*")
        return stop_event.set

    def _listen_motion():
        try:
            while not stop_event.is_set():
                pepper.setAngles(["HeadPitch"], [0.18], 0.25)
                time.sleep(0.35)
                pepper.setAngles(["HeadPitch"], [-0.03], 0.25)
                time.sleep(0.45)

            pepper.setAngles(["HeadPitch"], [0.0], 0.25)

        except Exception as e:
            print(f"[WARN] Listening gesture error: {e}")

    threading.Thread(target=_listen_motion, daemon=True).start()
    return stop_event.set

def robot_listen(prompt_text: str, pepper=None) -> str:
    """
    Speak prompt then listen for user response.
    Tries microphone + faster-whisper transcription first, falls back to keyboard.
    """
    robot_say(prompt_text, pepper)

    if STT_AVAILABLE and WHISPER_AVAILABLE:
        stop_listening_gesture = robot_listening_gesture(pepper)
        try:
            sample_rate = 16000
            phrase_seconds = 6
            print("[INFO] Listening via microphone...")
            samples = sd.rec(
                int(sample_rate * phrase_seconds),
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            stop_listening_gesture()

            samples = np.squeeze(samples)

            segments, _info = _whisper_model.transcribe(samples, language="en")
            text = " ".join(seg.text for seg in segments).strip()

            if not text:
                raise ValueError("empty transcription")

            print(f"[USER via mic/whisper] {text}")
            return text
        except Exception as e:
            try:
                stop_listening_gesture()
            except Exception:
                pass
            print(f"[INFO] Mic/Whisper failed ({e}) — switching to keyboard.")
    elif STT_AVAILABLE and not WHISPER_AVAILABLE:
        print("[INFO] Microphone present but faster-whisper missing — using keyboard.")

    answer = input("[YOU] ").strip()
    return answer

#  GESTURES  — all run in daemon threads so they don't block the main loop
def _run_in_thread(fn):
    """Helper: run fn in a daemon thread."""
    threading.Thread(target=fn, daemon=True).start()

_idle_motion_lock = threading.Lock()
_idle_motion_running = False

def robot_idle_patrol(pepper=None) -> None:
    '''Small idle movement while EventMate is waiting for a person'''
    global _idle_motion_running
    if not QIBULLET_AVAILABLE or pepper is None:
        print("[SOCIOBOT] *idle patrol movement*")
        return

    with _idle_motion_lock:
        if _idle_motion_running:
            return
        _idle_motion_running = True

    def _patrol():
        global _idle_motion_running
        try:
            action = random.choice(["turn_left", "turn_right", "small_forward"])

            if action == "turn_left":
                pepper.moveTo(0.0, 0.0, 0.35)
            elif action == "turn_right":
                pepper.moveTo(0.0, 0.0, -0.35)
            else:
                pepper.moveTo(0.20, 0.0, 0.0)

        except Exception as e:
            print(f"[WARN] Idle patrol error: {e}")
        finally:
            with _idle_motion_lock:
                _idle_motion_running = False

    threading.Thread(target=_patrol, daemon=True).start()


def robot_stop_motion(pepper=None) -> None:
    """
    Stop robot base motion when a user is detected.
    """
    if not QIBULLET_AVAILABLE or pepper is None:
        print("[SOCIOBOT] *stops moving*")
        return

    try:
        if hasattr(pepper, "stopMove"):
            pepper.stopMove()
        else:
            pepper.moveTo(0.0, 0.0, 0.0)
    except Exception as e:
        print(f"[WARN] Stop motion error: {e}")


# WAVE
def robot_wave(pepper=None) -> None:
    """Right-arm greeting wave."""
    if not QIBULLET_AVAILABLE or pepper is None:
        print("[PEPPER] *waves*")
        return

    def _wave():
        try:
            # 1. Raise arm, point upper arm right/forward, bend elbow 90 degrees (1.5 rad)
            pepper.setAngles(
                ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll"],
                [-1.5, -0.5, 0.0, 1.5], 
                0.5
            )
            time.sleep(0.6)
            
            # 2. Wave side-to-side by rotating the forearm (RElbowYaw)
            for _ in range(3):
                pepper.setAngles(["RElbowRoll"], [1.5], 0.6)  # Wave inward
                time.sleep(0.35)
                pepper.setAngles(["RElbowRoll"], [-0.5], 0.6) # Wave outward
                time.sleep(0.35)
                
            # 3. Return to resting posture
            pepper.setAngles(
                ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll"],
                [1.5, -0.1, 0.0, 0.5], 
                0.4
            )
        except Exception as e:
            print(f"[WARN] Wave error: {e}")

    _run_in_thread(_wave)


# NOD
def robot_nod(pepper=None, times: int = 2) -> None:
    """Head nod — acknowledgement."""
    if not QIBULLET_AVAILABLE or pepper is None:
        print("[SOCIOBOT] *nods*")
        return

    def _nod():
        try:
            for _ in range(times):
                pepper.setAngles(["HeadPitch"], [0.3], 0.1)
                time.sleep(0.45)
                pepper.setAngles(["HeadPitch"], [-0.01], 0.1)
                time.sleep(0.45)
            pepper.setAngles(["HeadPitch"], [0.0], 0.1)
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
        print("[SOCIOBOT] *thinking...*")
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


# PRESENTING 
def robot_present(pepper=None) -> None:
    """Presenting 'Ta-da!' gesture — open palm, used for top recommendations."""
    if not QIBULLET_AVAILABLE or pepper is None:
        print("[SOCIOBOT] *presents with a flourish!*")
        return

    def _present():
        try:
            # 1. Sweep right arm out and up, twist wrist to show open palm
            pepper.setAngles(
                ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand"],
                [0.2, -0.8, 1.5, 0.4, 1.0, 1.0], # Hand set to 1.0 (fully open)
                0.4
            )
            
            # Hold the pose so the user can look at the recommendation
            time.sleep(2.0)
            
            # 2. Reset back to a resting state
            pepper.setAngles(
                ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand"],
                [1.5, -0.1, 0.0, 0.5, 0.0, 1.0], 
                0.4
            )
        except Exception as e:
            print(f"[WARN] Presenting error: {e}")

    _run_in_thread(_present)


# EXCITED
def robot_excited(pepper=None) -> None:
    """Both-arms raise — used when greeting a known returning user."""
    if not QIBULLET_AVAILABLE or pepper is None:
        print("[SOCIOBOT] *excited!*")
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
        print("[SOCIOBOT] *farewell wave*")
        return

    def _farewell():
        try:
            # Raise arm high
            pepper.setAngles(
                ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RHand"],
                [-0.8, -0.5, 0.8, 1.0], 0.35
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
        print("[SOCIOBOT] *points forward*")
        return

    def _point():
        try:
            pepper.setAngles(
                ["RShoulderPitch", "RElbowRoll", "RElbowYaw", "RHand"],
                [-0.8, 0.3, 0.0, 0.0], 0.35
            )
            time.sleep(1.5)
            pepper.setAngles(
                ["RShoulderPitch", "RElbowRoll"],
                [1.5, 0.5], 0.3
            )
        except Exception as e:
            print(f"[WARN] Point error: {e}")

    _run_in_thread(_point)


class RobotInteractionLayer:
    """
    Object-oriented façade for simulation, speech, listening, and gestures.

    The module-level functions remain available for backwards compatibility;
    this class simply groups the same robot behaviours behind one object.
    """

    def __init__(self, pepper=None):
        self.pepper = pepper
        self.sim_manager = None

    def launch_simulation(self):
        self.pepper, self.sim_manager = launch_simulation()
        return self.pepper, self.sim_manager

    def say(self, text: str) -> None:
        robot_say(text, self.pepper)

    def act_and_say(
        self,
        action_fn,
        text: str,
        *,
        action_delay: float = 0.0,
        speech_delay: float = 0.25,
    ) -> None:
        robot_act_and_say(
            action_fn,
            text,
            self.pepper,
            action_delay=action_delay,
            speech_delay=speech_delay,
        )

    def listen(self, prompt_text: str) -> str:
        return robot_listen(prompt_text, self.pepper)

    def listening_gesture(self):
        return robot_listening_gesture(self.pepper)

    def idle_patrol(self) -> None:
        robot_idle_patrol(self.pepper)

    def stop_motion(self) -> None:
        robot_stop_motion(self.pepper)

    def wave(self) -> None:
        robot_wave(self.pepper)

    def nod(self, times: int = 2) -> None:
        robot_nod(self.pepper, times=times)

    def think(self, duration: float = 2.5) -> None:
        robot_thinking(self.pepper, duration=duration)

    def present(self) -> None:
        robot_present(self.pepper)

    def excited(self) -> None:
        robot_excited(self.pepper)

    def farewell(self) -> None:
        robot_farewell(self.pepper)

    def point(self) -> None:
        robot_point(self.pepper)

    def stop_simulation(self) -> None:
        if self.sim_manager is None:
            return
        manager, client_id = self.sim_manager
        manager.stopSimulation(client_id)


if __name__ == "__main__":
    
    from gestures import launch_simulation, robot_wave, robot_nod, robot_thinking, robot_present, robot_farewell
    import time

    pepper, sim_data = launch_simulation()
    sim_manager, client_id = sim_data
    robot_wave(pepper); time.sleep(3)
    robot_nod(pepper);  time.sleep(2)
    robot_thinking(pepper, 2.0); time.sleep(2)
    robot_present(pepper); time.sleep(3)
    robot_farewell(pepper);  time.sleep(3)

    sim_manager.stopSimulation(client_id)
