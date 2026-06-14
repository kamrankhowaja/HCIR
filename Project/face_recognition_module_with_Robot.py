from __future__ import annotations

"""
face_recognition_module.py
----------------------------
Face Recognition Module for the Preference-Based Social Event Recommendation Robot.
Integrated with qiBullet (Pepper simulation) for spoken interaction.

Robot behaviour:
  - Pepper simulation runs in qiBullet.
  - Webcam detects a face live.
  - KNOWN  -> Pepper waves + says "Welcome back, <name>!"
  - UNKNOWN -> Pepper says "Hi! I don't know you.", asks for name via TTS,
               user types name in terminal (or mic via SpeechRecognition),
               Pepper confirms registration and waves.

TTS  : pyttsx3  (offline, no internet needed)  -> pip install pyttsx3
STT  : SpeechRecognition + pyaudio (optional)  -> pip install SpeechRecognition pyaudio
       Falls back to terminal input() if mic not available.

Face : face_recognition + opencv-python + numpy

qiBullet : pip install qibullet
"""

import os
import sys
import time
import pickle
import threading

import cv2
import numpy as np
# pyrefly: ignore [missing-import]
import face_recognition
# pyrefly: ignore [missing-import]
import pyttsx3
# pyrefly: ignore [missing-import]
import speech_recognition as sr

# pyrefly: ignore [missing-import]
import pybullet as pb
# pyrefly: ignore [missing-import]
from qibullet import SimulationManager, PepperVirtual

# ── TTS setup ────────────────────────────────────────────────────────────────
try:
    
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty("rate", 150)   # speaking speed
    _tts_engine.setProperty("volume", 1.0)
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False
    print("[WARN] pyttsx3 not available. Install with: pip install pyttsx3")

# ── STT setup (optional) ─────────────────────────────────────────────────────
try:
    
    _recognizer  = sr.Recognizer()
    _microphone  = sr.Microphone()
    STT_AVAILABLE = True
except Exception:
    STT_AVAILABLE = False
    print("[INFO] SpeechRecognition/pyaudio not available. Using keyboard input.")

# ── qiBullet setup ───────────────────────────────────────────────────────────
try:
    QIBULLET_AVAILABLE = True
except Exception:
    QIBULLET_AVAILABLE = False
    print("[WARN] qiBullet not available. Robot gestures will be skipped.")

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR            = "known_faces"
DATA_FILE           = os.path.join(DATA_DIR, "face_data.pkl")
TOLERANCE           = 0.50
CONFIRMATION_FRAMES = 5
UNKNOWN_PATIENCE    = 5
NO_FACE_GRACE       = 15
# ─────────────────────────────────────────────────────────────────────────────

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)


# ════════════════════════════════════════════════════════════════════════════
#  ROBOT INTERACTION LAYER
# ════════════════════════════════════════════════════════════════════════════

def robot_say(text: str, pepper=None) -> None:
    """
    Make Pepper speak:
      1. Prints to terminal (always).
      2. Speaks aloud via pyttsx3 TTS (if available).
    In a real NAOqi setup replace pyttsx3 with ALTextToSpeech.say().
    """
    print("[PEPPER] " + text)
    if TTS_AVAILABLE:
        try:
            _tts_engine.say(text)
            _tts_engine.runAndWait()
        except Exception as e:
            print("[WARN] TTS error: {}".format(e))


def robot_wave(pepper=None) -> None:
    """
    Make the simulated Pepper wave its right arm as a greeting gesture.
    Runs in a background thread so it does not block recognition.
    """
    if not QIBULLET_AVAILABLE or pepper is None:
        print("[PEPPER] *waves*")
        return

    def _wave():
        try:
            # Raise right arm
            pepper.setAngles(
                ["RShoulderPitch", "RShoulderRoll", "RElbowRoll"],
                [-0.5,            -0.3,             1.2],
                0.4
            )
            time.sleep(1.2)
            # Wave left-right
            for _ in range(2):
                pepper.setAngles(["RShoulderRoll"], [-0.8], 0.6)
                time.sleep(0.4)
                pepper.setAngles(["RShoulderRoll"], [-0.1], 0.6)
                time.sleep(0.4)
            # Return to rest
            pepper.setAngles(
                ["RShoulderPitch", "RShoulderRoll", "RElbowRoll"],
                [1.5,              -0.1,             0.5],
                0.3
            )
        except Exception as e:
            print("[WARN] Wave gesture error: {}".format(e))

    threading.Thread(target=_wave, daemon=True).start()


def robot_nod(pepper=None) -> None:
    """Small head nod to acknowledge the user."""
    if not QIBULLET_AVAILABLE or pepper is None:
        print("[PEPPER] *nods*")
        return

    def _nod():
        try:
            pepper.setAngles(["HeadPitch"], [0.3], 0.3)
            time.sleep(0.5)
            pepper.setAngles(["HeadPitch"], [0.0], 0.3)
        except Exception as e:
            print("[WARN] Nod gesture error: {}".format(e))

    threading.Thread(target=_nod, daemon=True).start()


def robot_listen(prompt_text: str, pepper=None) -> str:
    """
    Ask a question via TTS, then listen for the answer.
    Priority:
      1. SpeechRecognition + microphone (if available)
      2. keyboard input() fallback
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
            print("[USER via mic] " + text)
            return text.strip()
        except Exception as e:
            print("[INFO] Mic recognition failed ({}). Switching to keyboard.".format(e))

    # Fallback: keyboard
    answer = input("[YOU] ").strip()
    return answer


# ════════════════════════════════════════════════════════════════════════════
#  FACE DATABASE HELPERS
# ════════════════════════════════════════════════════════════════════════════

def load_known_faces() -> dict:
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_known_faces(db: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DATA_FILE, "wb") as f:
        pickle.dump(db, f)


def identify_face(encoding, db: dict):
    best_name = None
    best_dist = float("inf")
    for name, stored in db.items():
        distances = face_recognition.face_distance(stored, encoding)
        d = float(np.min(distances))
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name if best_dist <= TOLERANCE else None


def quick_face_present(gray_frame) -> bool:
    faces = _face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
    )
    return len(faces) > 0


# ════════════════════════════════════════════════════════════════════════════
#  REGISTRATION FLOW  (with robot speech)
# ════════════════════════════════════════════════════════════════════════════

def register_new_user(encoding, db: dict, pepper=None) -> str:
    """Full spoken registration flow via Pepper."""

    robot_say("Hello there! I have not seen you before.", pepper)
    time.sleep(0.3)

    name = robot_listen("What is your name, please?", pepper)

    if not name:
        name = "Guest"

    # Confirm back
    robot_say("Nice to meet you, {}!".format(name), pepper)
    time.sleep(0.2)
    robot_say("I will remember your face for next time.", pepper)
    robot_wave(pepper)

    # Save to database
    if name not in db:
        db[name] = []
    db[name].append(encoding)
    save_known_faces(db)

    print("[INFO] User '{}' registered and saved.".format(name))
    return name


# ════════════════════════════════════════════════════════════════════════════
#  MAIN RECOGNITION LOOP
# ════════════════════════════════════════════════════════════════════════════

def run_face_recognition(pepper=None) -> tuple:
    """
    Live webcam face recognition with full robot interaction.

    Parameters
    ----------
    pepper : PepperVirtual instance (from qiBullet) or None

    Returns
    -------
    (status, name)
      status = "known"   -> returning user
      status = "new"     -> just registered
      status = "unknown" -> quit without recognition
    """
    db  = load_known_faces()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    confirm_buffer  = {}
    unknown_counter = 0
    no_face_frames  = 0
    last_enc        = None
    result          = None

    robot_say("Hello! Please look at the camera so I can recognise you.", pepper)
    print("[INFO] Face recognition started. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, locations)
        locations_full = [(t*2, r*2, b*2, l*2) for t, r, b, l in locations]

        if not encodings:
            if quick_face_present(gray):
                no_face_frames = 0   # Haar still sees something — grace
            else:
                no_face_frames += 1
                if no_face_frames >= NO_FACE_GRACE:
                    unknown_counter = 0
                    confirm_buffer  = {}
                    last_enc        = None
                    no_face_frames  = 0

            cv2.putText(frame, "Looking for face...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        else:
            no_face_frames = 0

            for enc, (top, right, bottom, left) in zip(encodings, locations_full):
                last_enc     = enc
                matched_name = identify_face(enc, db)

                if matched_name:
                    # ── Known user ────────────────────────────────────────
                    unknown_counter = 0
                    confirm_buffer[matched_name] = confirm_buffer.get(matched_name, 0) + 1

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 0), 2)
                    cv2.putText(
                        frame,
                        "{} ({}/{})".format(matched_name, confirm_buffer[matched_name], CONFIRMATION_FRAMES),
                        (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2
                    )

                    if confirm_buffer[matched_name] >= CONFIRMATION_FRAMES:
                        cap.release()
                        cv2.destroyAllWindows()
                        # Robot greeting
                        robot_wave(pepper)
                        robot_say("Welcome back, {}! Great to see you again.".format(matched_name), pepper)
                        robot_nod(pepper)
                        return ("known", matched_name)

                else:
                    # ── Unknown user ──────────────────────────────────────
                    unknown_counter += 1
                    confirm_buffer   = {}

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 220), 2)
                    cv2.putText(
                        frame,
                        "Unknown ({}/{})".format(unknown_counter, UNKNOWN_PATIENCE),
                        (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2
                    )

                    if unknown_counter >= UNKNOWN_PATIENCE:
                        cap.release()
                        cv2.destroyAllWindows()
                        enc_to_save = last_enc if last_enc is not None else enc
                        name = register_new_user(enc_to_save, db, pepper)
                        return ("new", name)

        cv2.imshow("Pepper - Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return ("unknown", "unidentified")


# ════════════════════════════════════════════════════════════════════════════
#  QIBULLET SIMULATION LAUNCHER
# ════════════════════════════════════════════════════════════════════════════

def launch_simulation():
    """
    Starts the qiBullet simulation, loads Pepper, and returns the pepper instance.
    Returns None if qiBullet is not installed.
    """
    if not QIBULLET_AVAILABLE:
        print("[INFO] qiBullet not available — running without simulation.")
        return None

    sim_manager = SimulationManager()
    client_id   = sim_manager.launchSimulation(gui=True)
    pepper      = sim_manager.spawnPepper(client_id, spawn_ground_plane=True)

    # Initial idle pose
    pepper.goToPosture("StandInit", 0.6)
    time.sleep(1.5)
    return pepper


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ════════════════════════════════════════════════════════════════════════════

def get_user_status(pepper=None) -> tuple:
    """
    Main entry point for the robot pipeline.

    Parameters
    ----------
    pepper : PepperVirtual or None

    Returns
    -------
    (status, name)
      "known"   -> returning user  → skip preference survey
      "new"     -> new user        → run preference survey
      "unknown" -> unresolved      → handle gracefully
    """
    return run_face_recognition(pepper)


def list_registered_users() -> list:
    return list(load_known_faces().keys())


def delete_user(name: str) -> bool:
    db = load_known_faces()
    if name in db:
        del db[name]
        save_known_faces(db)
        print("[INFO] User '{}' deleted.".format(name))
        return True
    print("[INFO] User '{}' not found.".format(name))
    return False


# ════════════════════════════════════════════════════════════════════════════
#  STANDALONE ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pepper = launch_simulation()           # starts qiBullet with Pepper
    status, name = get_user_status(pepper) # run face recognition
    print("\nResult -> status={!r}, name={!r}".format(status, name))
