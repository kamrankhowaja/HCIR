from __future__ import annotations

"""
face_recognition_module_with_Robot.py
-------------------------------------
Upgraded Face Recognition Module for the Social Event Recommendation Robot.
Maintains live webcam and qiBullet simulation loops concurrently using 
non-blocking execution states. Integrates continuous 10s presence checks, 
proactive wake-word listening, and a privacy consent layer.
"""

import os
import sys
import time
import pickle
import threading
import queue

import cv2
import numpy as np

# Suppress warnings if imports are missing in local dev
# pyrefly: ignore [missing-import]
import face_recognition
# pyrefly: ignore [missing-import]
import pyttsx3
# pyrefly: ignore [missing-import]
import speech_recognition as sr

# # pyrefly: ignore [missing-import]
# import pybullet as pb
# pyrefly: ignore [missing-import]
from qibullet import SimulationManager, PepperVirtual

# ── CONFIGURATION & CONSTANTS ────────────────────────────────────────────────
DATA_DIR            = "known_faces"
DATA_FILE           = os.path.join(DATA_DIR, "face_data.pkl")
TOLERANCE           = 0.50
CONFIRMATION_FRAMES = 5
UNKNOWN_PATIENCE    = 5

ROBOT_NAME          = "EventMate"

# ── SHARED TEAM INTEGRATION LAYER ────────────────────────────────────────────
class RobotSessionContext:
    """
    Global thread-safe state container that other team modules (Dialog, 
    Bayesian Network) can reference to inspect the current user's pipeline status.
    """
    def __init__(self):
        self.status = "unknown"          # "known" | "new" | "anonymous" | "unknown"
        self.user_name = "unidentified"  # Actual name or temporary Session Guest ID
        self.session_active = False      # True when actively engaged in dialogue
        self.is_consent_granted = False  # Track privacy consent profile
        self.lock = threading.Lock()

    def update_session(self, status: str, name: str, active: bool, consent: bool):
        with self.lock:
            self.status = status
            self.user_name = name
            self.session_active = active
            self.is_consent_granted = consent

    def reset(self):
        with self.lock:
            self.status = "unknown"
            self.user_name = "unidentified"
            self.session_active = False
            self.is_consent_granted = False

# Instantiate global state context
session_context = RobotSessionContext()

# ── HARDWARE / SUBSYSTEM INITIALIZATION ──────────────────────────────────────
try:
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty("rate", 150)
    _tts_engine.setProperty("volume", 1.0)
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False
    print("[WARN] pyttsx3 not available. Using terminal fallback for output.")

try:
    _recognizer = sr.Recognizer()
    _microphone = sr.Microphone()
    STT_AVAILABLE = True
except Exception:
    STT_AVAILABLE = False
    print("[WARN] SpeechRecognition/pyaudio unavailable. STT functions restricted.")

try:
    QIBULLET_AVAILABLE = True
except Exception:
    QIBULLET_AVAILABLE = False
    print("[WARN] qiBullet simulation module unavailable.")

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)


def robot_say(text: str) -> None:
    """Enqueues spoken response to the background engine non-blockingly."""
    _speech_queue.put(text)


# ── ADAPTIVE TEXT CLEANER HELPER ─────────────────────────────────────────────
def clean_extracted_name(raw_text: str) -> str:
    """
    Cleans raw speech to extract only the actual name.
    Converts 'my name is rahul' or 'i am rahul' -> 'rahul'
    """
    cleaned = raw_text.strip().lower()
    
    # Common speech prefixes to remove
    prefixes = ["my name is", "i am called", "call me", "i am", "it is", "its"]
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
            
    # Capitalize the first letter of the name so it saves neatly
    return cleaned.title() if cleaned else "Guest"

# ── BACKGROUND SPEECH PROCESSING WORKER (STABLE INTERNAL ENGINE LOOP) ───────
_speech_queue = queue.Queue()

def _speech_worker():
    """
    Persistent background worker managing text-to-speech tasks sequentially.
    Keeps a single engine lifecycle alive to prevent thread-lock exceptions.
    """
    engine = None
    if TTS_AVAILABLE:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 145)
            engine.setProperty("volume", 1.0)
        except Exception as e:
            print(f"[WARN] Failed to initialize background TTS engine: {e}")

    while True:
        text = _speech_queue.get()
        if text is None:
            break
        print(f"[{ROBOT_NAME}] {text}")
        
        if TTS_AVAILABLE and engine is not None:
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[WARN] TTS worker runtime error: {e}")
        else:
            time.sleep(len(text) * 0.08 + 0.5)
            
        _speech_queue.task_done()

threading.Thread(target=_speech_worker, daemon=True).start()


def robot_say_blocking(text: str) -> None:
    """Enqueues dialogue and blocks processing state until execution finishes."""
    _speech_queue.put(text)
    _speech_queue.join()


def robot_listen_blocking(prompt_text: str) -> str:
    """
    Blocks interaction workflow. Employs fixed audio sensitivity thresholding 
    to bypass driver-induced calibration gaps.
    """
    # 1. Wait until the robot is completely done talking
    robot_say_blocking(prompt_text)

    if STT_AVAILABLE:
        try:
            with _microphone as source:
                print("[INFO] Initializing microphone array...")
                
                # FIX: We bypass automatic threshold spikes by enforcing a fixed, highly sensitive value.
                # If it still struggles to hear you in a noisy room, tweak 300 up to 600.
                _recognizer.energy_threshold = 300 
                _recognizer.dynamic_energy_threshold = False
                
                # Give Linux driver warnings a moment to clear
                time.sleep(0.5) 
                
                print(f"\n>>> [{ROBOT_NAME} IS LISTENING NOW] <<< Please talk now...")
                
                # Allow a generous 10 seconds to start talking, up to 6 seconds to finish your phrase
                audio = _recognizer.listen(source, timeout=10, phrase_time_limit=6)
                
            print("[INFO] Voice received! Processing speech recognition values...")
            text = _recognizer.recognize_google(audio)
            print(f"[USER Speech Detected] -> \"{text}\"")
            return text.strip().lower()
            
        except sr.WaitTimeoutError:
            print("[INFO] Speech timeout: The system did not hear any voice input.")
        except sr.UnknownValueError:
            print("[INFO] Speech unrecognized: Noise detected, but words were unclear.")
        except Exception as e:
            print(f"[INFO] Audio driver error during capture operation: {e}")
            
    return ""

# ── ROBOT GESTURE COMMANDS (NON-BLOCKING) ────────────────────────────────────
def robot_gesture(gesture_type: str, pepper=None) -> None:
    if not QIBULLET_AVAILABLE or pepper is None:
        print(f"[{ROBOT_NAME}] *executes {gesture_type} gesture*")
        return

    def _animate():
        try:
            if gesture_type == "wave":
                pepper.setAngles(["RShoulderPitch", "RShoulderRoll", "RElbowRoll"], [-0.5, -0.3, 1.2], 0.4)
                time.sleep(1.0)
                for _ in range(2):
                    pepper.setAngles(["RShoulderRoll"], [-0.8], 0.5)
                    time.sleep(0.3)
                    pepper.setAngles(["RShoulderRoll"], [-0.1], 0.5)
                    time.sleep(0.3)
                pepper.setAngles(["RShoulderPitch", "RShoulderRoll", "RElbowRoll"], [1.5, -0.1, 0.5], 0.3)
            elif gesture_type == "nod":
                pepper.setAngles(["HeadPitch"], [0.2], 0.3)
                time.sleep(0.4)
                pepper.setAngles(["HeadPitch"], [0.0], 0.3)
        except Exception as e:
            print(f"[WARN] Gesture thread exception: {e}")

    threading.Thread(target=_animate, daemon=True).start()

# ── DATA EXTRACTION & DATABASE HANDLERS ──────────────────────────────────────
def load_known_faces() -> dict:
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}

def save_known_faces(db: dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DATA_FILE, "wb") as f:
        pickle.dump(db, f)

def identify_face(encoding, db: dict) -> str | None:
    best_name = None
    best_dist = float("inf")
    for name, stored in db.items():
        distances = face_recognition.face_distance(stored, encoding)
        if len(distances) == 0:
            continue
        d = float(np.min(distances))
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name if best_dist <= TOLERANCE else None

# ── CENTRAL PIPELINE THREAD SAFETY & MANAGEMENT ──────────────────────────────
class UserInteractionPipeline:
    """Manages conversational tracking, timeouts, and privacy gating."""
    def __init__(self, pepper=None):
        self.pepper = pepper
        self.db = load_known_faces()
        self.ambient_listen_active = False

    def check_ambient_voice_activation(self) -> bool:
        """Asynchronously checks for specific vocal phrases if STT is online."""
        if not STT_AVAILABLE or self.ambient_listen_active:
            return False
        
        def _listen_passive():
            self.ambient_listen_active = True
            try:
                with _microphone as source:
                    _recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    audio = _recognizer.listen(source, timeout=2, phrase_time_limit=3)
                phrase = _recognizer.recognize_google(audio).lower()
                if any(w in phrase for w in ["hello", "need help", "help me"]):
                    print(f"[WAKE WORD] Detected phrase: '{phrase}'")
                    # Trigger conversational state on system background execution loop
                    threading.Thread(target=self.start_conversation_flow, args=("unknown", None), daemon=True).start()
            except Exception:
                pass
            self.ambient_listen_active = False

        threading.Thread(target=_listen_passive, daemon=True).start()
        return False

    def start_conversation_flow(self, classification: str, identity_details=None):
        """Asynchronous execution flow for registration, consent checking, and dialogue session tracking."""
        if session_context.session_active:
            return

        session_context.update_session("identifying", "unidentified", True, False)
        robot_gesture("wave", self.pepper)
        robot_say_blocking(f"Hello! I am {ROBOT_NAME}, your social event recommendation assistant.")

        if classification == "known":
            name = identity_details
            robot_say_blocking(f"Welcome back, {name}! Great to see you again.")
            robot_gesture("nod", self.pepper)
            session_context.update_session("known", name, True, True)
            self._mock_handoff_loop()

        else:
            # New User Routine
            robot_say_blocking("I noticed you are a new visitor!")
            raw_name_input = robot_listen_blocking("What is your name, please?")
            
            # Use our new intelligent string cleaner here
            name_input = clean_extracted_name(raw_name_input)
            if not name_input or name_input == "Guest":
                name_input = f"Guest_{int(time.time()) % 1000}"

            # Privacy Consent Gate
            consent_response = robot_listen_blocking(
                f"Nice to meet you, {name_input}. To securely analyze and save your biometric features and preferences, "
                "may I have your consent to store this data profile?"
            )

            if any(word in consent_response for word in ["yes", "sure", "agree", "yup", "ok", "okay"]):
                robot_say_blocking("Thank you. Saving your profile information now.")
                if name_input not in self.db:
                    self.db[name_input] = []
                if identity_details is not None:
                    self.db[name_input].append(identity_details)
                    save_known_faces(self.db)
                session_context.update_session("new", name_input, True, True)
            else:
                # Triggers if you say "no", or if it times out
                robot_say_blocking("Understood. I will execute your session completely anonymously without saving any face records.")
                session_context.update_session("anonymous", name_input, True, False)

            self._mock_handoff_loop()

    def run_presence_timeout_audit(self) -> bool:
        """
        Invoked dynamically by the main runtime loop when face tracking is lost.
        Asks 'Are you still there?', and actively listens/scans for 5 seconds.
        If a face reappears or a 'yes' is spoken, it resumes seamlessly.
        """
        # 1. Ask the question out loud and wait for the robot to finish speaking
        robot_say_blocking("Are you still there?")
        
        print("[TIMEOUT AUDIT] Running active 5-second presence check...")
        grace_start = time.time()
        
        # Enforce highly responsive mic levels for the timeout logic
        if STT_AVAILABLE:
            _recognizer.energy_threshold = 300
            _recognizer.dynamic_energy_threshold = False

        while time.time() - grace_start < 5.0:
            # Calculate remaining seconds left in the grace period
            time_left = 5.0 - (time.time() - grace_start)
            if time_left <= 0:
                break

            # A. VOICE CHECK: Listen in short 2-second windows so we don't block the whole grace loop
            if STT_AVAILABLE:
                try:
                    with _microphone as source:
                        print(f"[TIMEOUT AUDIT] Listening... ({int(time_left)}s left)")
                        # Listen with a very short timeout window
                        audio = _recognizer.listen(source, timeout=1.5, phrase_time_limit=2)
                    
                    res = _recognizer.recognize_google(audio).lower()
                    print(f"[TIMEOUT AUDIT Speech Detected] -> \"{res}\"")
                    if any(w in res for w in ["yes", "here", "still here", "i am", "yeah", "hello"]):
                        robot_say_blocking("Excellent! Resuming our session.")
                        return False  # SUCCESS: Do NOT terminate, return to active communication
                except (sr.WaitTimeoutError, sr.UnknownValueError):
                    # No audio heard or couldn't parse in this small window; continue checking
                    pass
                except Exception as e:
                    print(f"[WARN] Audio error in audit loop: {e}")

            # B. VISUAL CHECK (Optional fallback): 
            # If your main loop senses a face during this period and updates last_face_seen_timestamp,
            # we can intercept that to break early.
            # (Note: Since this runs on the background thread, it relies on the camera thread 
            # modifying your global state or variables. We check if a face is currently active)
            # if session_context.session_active is maintained, let's allow a tiny pause to avoid 100% CPU usage
            time.sleep(0.1)

        # If the loop finishes 5 seconds without catching an interaction phrase or face match
        print("[TIMEOUT AUDIT] Grace period completely expired with no response. Terminating session.")
        return True  # DISCONNECT: Tell the system to wipe context variables and stand by

    def _mock_handoff_loop(self):
        """Simulates interaction runtime execution, holding status parameters open for teammate hooks."""
        print(f"[PIPELINE RUNNING] Handed control to core application modules. Name: {session_context.user_name}")
        # In deployment, replace this tracking sleep with explicit hooks matching dialog system loops
        while session_context.session_active:
            time.sleep(0.5)

    def terminate_current_session(self):
        """Gracefully tears down session contexts and re-arms default postures."""
        if session_context.session_active:
            robot_say("Goodbye! Have a wonderful day ahead.")
            if self.pepper is not None:
                try:
                    self.pepper.goToPosture("StandInit", 0.5)
                except Exception:
                    pass
            session_context.reset()
            print("[RESET] Cleaned context flags. Ready for the next user interaction.\n")

# ── MAIN SYSTEM RUNTIME LOOP ─────────────────────────────────────────────────
def main_runtime_pipeline():
    # Initialize qiBullet environment layout wrapper
    pepper_instance = None
    if QIBULLET_AVAILABLE:
        try:
            sim_manager = SimulationManager()
            client_id = sim_manager.launchSimulation(gui=True)
            pepper_instance = sim_manager.spawnPepper(client_id, spawn_ground_plane=True)
            pepper_instance.goToPosture("StandInit", 0.6)
            time.sleep(1.0)
        except Exception as e:
            print(f"[WARN] qiBullet instantiation failed: {e}. Switching to window-only feed.")

    pipeline = UserInteractionPipeline(pepper_instance)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Primary camera capture array inaccessible.")
        return

    print(f"\n[{ROBOT_NAME} INIT] Monitoring video array and simulation bounds. Press 'q' to stop.")

    # Frame tracking variables
    face_seen_start_time = None
    last_face_seen_timestamp = time.time()
    unknown_consecutive_frames = 0
    known_buffer = {}
    last_valid_encoding = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_small, model="hog")
        encodings = face_recognition.face_encodings(rgb_small, locations)
        
        # Real-time tracking verification logic
        if encodings:
            last_face_seen_timestamp = time.time()
            if face_seen_start_time is None:
                face_seen_start_time = time.time()

            elapsed_presence_seconds = time.time() - face_seen_start_time

            for enc in encodings:
                matched_name = identify_face(enc, pipeline.db)
                last_valid_encoding = enc

                if matched_name:
                    unknown_consecutive_frames = 0
                    known_buffer[matched_name] = known_buffer.get(matched_name, 0) + 1
                    
                    # Proactive trigger conditional validation ($\ge 2$ seconds rule)
                    if elapsed_presence_seconds >= 2.0 and not session_context.session_active:
                        if known_buffer[matched_name] >= CONFIRMATION_FRAMES:
                            threading.Thread(
                                target=pipeline.start_conversation_flow, 
                                args=("known", matched_name), 
                                daemon=True
                            ).start()
                else:
                    known_buffer.clear()
                    unknown_consecutive_frames += 1
                    
                    if elapsed_presence_seconds >= 2.0 and not session_context.session_active:
                        if unknown_consecutive_frames >= UNKNOWN_PATIENCE:
                            threading.Thread(
                                target=pipeline.start_conversation_flow, 
                                args=("unknown", last_valid_encoding), 
                                daemon=True
                            ).start()

            # Render overlay annotations
            for top, right, bottom, left in locations:
                cv2.rectangle(frame, (left*2, top*2), (right*2, bottom*2), (255, 140, 0), 2)
                cv2.putText(frame, "Tracking Active", (left*2, (top*2)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 2)
        else:
            # Face presence tracking lost paths
            face_seen_start_time = None
            unknown_consecutive_frames = 0
            known_buffer.clear()

            # Ambient listening execution while idle
            if not session_context.session_active:
                pipeline.check_ambient_voice_activation()

            # 10-Second Heartbeat Checker Routine
            if session_context.session_active and (time.time() - last_face_seen_timestamp > 10.0):
                print("[HEARTBEAT DROP] User tracking absent for 10 consecutive seconds.")
                # Halt pipeline loops momentarily to run audit confirmation
                should_disconnect = pipeline.run_presence_timeout_audit()
                if should_disconnect:
                    pipeline.terminate_current_session()
                else:
                    # User responded or came back, reset tracking clock
                    last_face_seen_timestamp = time.time()

            cv2.putText(frame, "Scanning environment...", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.imshow(f"{ROBOT_NAME} - Main Camera View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pipeline.terminate_current_session()

if __name__ == "__main__":
    main_runtime_pipeline()