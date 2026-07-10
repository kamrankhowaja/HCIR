"""
Face Detection and Recognition module for SocioBot
"""
from __future__ import annotations
from cv2 import CAP_V4L2

import importlib
import os
import re
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def _import_face_recognition_library():
    """
    Import the third-party face_recognition package without shadowing this file.
    to avoid accidentally importing this local module instead of the installed package.
    """
    module_path = Path(__file__).resolve()
    module_dir = module_path.parent
    original_sys_path = list(sys.path)

    try:
        sys.path = [
            entry for entry in sys.path
            if Path(entry or os.getcwd()).resolve() != module_dir
        ]
        library = importlib.import_module("face_recognition")
    except ImportError as exc:
        raise ImportError(
            "Could not import the third-party 'face_recognition' package. "
            "Install it with: pip install face-recognition"
        ) from exc
    finally:
        sys.path = original_sys_path

    library_path = getattr(library, "__file__", None)
    if library_path and Path(library_path).resolve() == module_path:
        raise ImportError(
            "Imported this local module instead of the third-party "
            "'face_recognition' package. Run from the project root or install "
            "the package with: pip install face-recognition"
        )

    missing = [name for name in ("face_locations", "face_encodings", "face_distance")
               if not hasattr(library, name)]
    if missing:
        raise ImportError(
            "The imported 'face_recognition' module is missing expected API(s): "
            + ", ".join(missing)
        )

    return library


fr = _import_face_recognition_library()

try:
    from modules.gestures import (
        robot_act_and_say, robot_say, robot_wave, robot_nod, robot_listen,
        robot_idle_patrol, robot_stop_motion
    )
except ModuleNotFoundError:
    from gestures import (
        robot_act_and_say, robot_say, robot_wave, robot_nod, robot_listen,
        robot_idle_patrol, robot_stop_motion
    )
# Config
DATA_DIR            = "./known_faces"
DATA_FILE           = os.path.join(DATA_DIR, "face_data.pkl")

TOLERANCE           = 0.50          # lower = stricter (0.4–0.6 typical)
CONFIRMATION_FRAMES = 5             # consecutive matched frames to confirm identity
UNKNOWN_PATIENCE    = 15            # unknown frames before triggering registration
NO_FACE_GRACE       = 15            # no-face frames before resetting counters
IDLE_PATROL_INTERVAL_SEC = 4.0      # seconds between small idle movements
_PENDING_NEW_USER_ENCODING = None

# Haar cascade for fast pre-screening
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)


class FaceDatabase:
    """Handles loading, saving, and updating known-face data."""

    def __init__(self, data_dir: str = DATA_DIR, data_file: str = DATA_FILE):
        self.data_dir = data_dir
        self.data_file = data_file

    def load(self) -> dict:
        """Load {name: [encodings]} from disk. Returns empty dict if not found."""
        os.makedirs(self.data_dir, exist_ok=True)
        if os.path.exists(self.data_file):
            with open(self.data_file, "rb") as f:
                return pickle.load(f)
        return {}

    def save(self, db: dict) -> None:
        """Persist face database to disk."""
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.data_file, "wb") as f:
            pickle.dump(db, f)

    @staticmethod
    def get_user_encodings(entry) -> list:
        """
        Return the stored face encodings for one database entry.

        Supports both database formats:
          - legacy: db[name] = [encoding, ...]
          - current: db[name] = {"encodings": [...], "last_interest": "..."}
        """
        if isinstance(entry, dict):
            return entry.get("encodings", [])
        if isinstance(entry, list):
            return entry
        return []

    def get_preference_history(self, name: str) -> str:
        """
        Return the most recent interest category stored for a known user,
        or 'Unknown' if the user has no stored preference.
        """
        db = self.load()
        entry = db.get(name, None)

        if entry is None:
            return "Unknown"

        if isinstance(entry, dict):
            return entry.get("last_interest", "Unknown")

        return "Unknown"

    def save_user_preference(self, name: str, interest: str) -> None:
        """Persist the user's last interest category after a session."""
        db = self.load()
        entry = db.get(name, None)

        if entry is None:
            return

        if isinstance(entry, list):
            encodings = entry
            db[name] = {"encodings": encodings, "last_interest": interest}
        elif isinstance(entry, dict):
            entry["last_interest"] = interest

        self.save(db)
        print(f"[INFO] Saved preference '{interest}' for user '{name}'.")

    def list_registered_users(self) -> list[str]:
        return list(self.load().keys())

    def delete_user(self, name: str) -> bool:
        db = self.load()
        if name in db:
            del db[name]
            self.save(db)
            print(f"[INFO] User '{name}' deleted.")
            return True
        print(f"[INFO] User '{name}' not found.")
        return False


DEFAULT_FACE_DATABASE = FaceDatabase()


#  PERSISTENCE
def load_known_faces() -> dict:
    return DEFAULT_FACE_DATABASE.load()


def save_known_faces(db: dict) -> None:
    DEFAULT_FACE_DATABASE.save(db)


def get_user_encodings(entry) -> list:
    return FaceDatabase.get_user_encodings(entry)


#  RECOGNITION HELPERS
def identify_face(encoding: np.ndarray, db: dict) -> Optional[str]:
    """
    Compare encoding against stored users.
    Returns best-matching name or None if no match within tolerance.
    """
    best_name = None
    best_dist = float("inf")

    for name, entry in db.items():
        stored_encodings = get_user_encodings(entry)
        if not stored_encodings:
            continue

        distances = fr.face_distance(stored_encodings, encoding)
        min_dist  = float(np.min(distances))
        if min_dist < best_dist:
            best_dist = min_dist
            best_name = name

    return best_name if best_dist <= TOLERANCE else None


def quick_face_present(gray_frame: np.ndarray) -> bool:
    """
    Fast Haar cascade check - True if any face visible.
    Used as a grace-period guard so a single missed frame doesn't reset counters.
    """
    faces = _face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
    )
    return len(faces) > 0


def get_preference_history(name: str) -> str:
    return DEFAULT_FACE_DATABASE.get_preference_history(name)


def save_user_preference(name: str, interest: str) -> None:
    DEFAULT_FACE_DATABASE.save_user_preference(name, interest)

def has_pending_new_user_encoding() -> bool:
    return _PENDING_NEW_USER_ENCODING is not None


def save_pending_new_user(name: str, interest: str = "Unknown") -> bool:
    """
    Save the temporarily stored face encoding only after end-of-interaction consent.
    """
    global _PENDING_NEW_USER_ENCODING

    if _PENDING_NEW_USER_ENCODING is None:
        print("[INFO] No pending face encoding to save.")
        return False

    if not name or name in ("Guest", "unidentified"):
        print("[INFO] No reliable name available. Face data not saved.")
        _PENDING_NEW_USER_ENCODING = None
        return False

    db = load_known_faces()

    if name not in db:
        db[name] = {"encodings": [], "last_interest": interest}

    entry = db[name]
    if isinstance(entry, list):
        entry.append(_PENDING_NEW_USER_ENCODING)
        db[name] = {"encodings": entry, "last_interest": interest}
    else:
        entry["encodings"].append(_PENDING_NEW_USER_ENCODING)
        entry["last_interest"] = interest

    save_known_faces(db)
    _PENDING_NEW_USER_ENCODING = None

    print(f"[INFO] Saved new user '{name}' after consent.")
    return True

def discard_pending_new_user_encoding() -> None:
    """
    Forget temporary face encoding if the user does not consent.
    """
    global _PENDING_NEW_USER_ENCODING
    _PENDING_NEW_USER_ENCODING = None
    print("[INFO] Pending face encoding discarded.")

def _normalize_answer(answer: str) -> str:
    return re.sub(r"[^a-zA-Z'\s]", " ", answer.lower()).strip()


def _contains_phrase(text: str, phrases: list[str]) -> bool:
    text = f" {text} "
    return any(f" {phrase} " in text for phrase in phrases)


def _is_yes(answer: str) -> bool:
    a = _normalize_answer(answer)
    return _contains_phrase(a, [
        "yes", "yeah", "yep", "sure", "okay",
        "go ahead", "that's fine", "that is fine",
        "you can", "i agree", "save it"
    ])


def _is_no(answer: str) -> bool:
    a = _normalize_answer(answer)
    return _contains_phrase(a, [
        "no", "nope", "do not", "don't", "dont",
        "not okay", "not fine", "rather not",
        "do not save", "don't save", "privacy"
    ])


def ask_face_storage_consent(pepper=None) -> bool:
    """
    Ask explicit consent before saving face data.
    Defaults to False if the answer is unclear.
    """
    prompts = [
        "Hello, I am SocioBot. I help Homo sapiens choose events based on their mood, budget, group size, and preferences. I have not seen you before. Is it okay if I save your face data and name so I can recognize you next time?",
        "Before I store anything, I need a clear yes or no. May I save your face data and name for future personalized recommendations?"
    ]

    for attempt, prompt in enumerate(prompts):
        answer = robot_listen(prompt, pepper).strip()

        if "interrupt" in answer.lower():
            raise KeyboardInterrupt("User interrupted during face-data consent.")

        if _is_no(answer):
            return False

        if _is_yes(answer):
            return True

        if attempt == 0:
            robot_say(
                "I just need a clear yes or no before saving any face data.",
                pepper
            )

    robot_say(
        "No problem. I will not save your face data. We can continue as a guest.",
        pepper
    )
    return False

def clean_spoken_name(raw_name: str) -> str:
    """
    Clean noisy Whisper name transcriptions.
    Keeps simple name-like answers and rejects obvious non-names.
    """
    text = raw_name.strip()

    parts = [p.strip() for p in re.split(r"[.!?,]", text) if p.strip()]
    if parts:
        text = parts[-1]

    text = re.sub(r"[^a-zA-Z'\-\s]", "", text).strip()

    filler_words = {
        "uh", "um", "actually", "please", "thanks", "thank", "you",
        "yes", "yeah", "sure", "okay", "great", "again", "once", "can",
        "going", "to", "go", "call", "me", "just", "put", "but", "my name is", "i am called", "call me", "i am", "it is", "its"
    }

    words = [w for w in text.split() if w.lower() not in filler_words]

    if not words:
        return "Guest"

    if len(words) > 3:
        return "Guest"

    name = " ".join(words).title()

    bad_names = {
        "But", "Again", "Once Again", "Thank You", "Great",
        "Yes", "No", "Okay", "Sure", "Just Put"
    }

    if name in bad_names:
        return "Guest"

    return name

def ask_temporary_name(prompt: str, pepper=None) -> str:
    """
    Ask for the user's name without a confirmation loop.
    Rephrases once if the transcription is unclear.
    """
    prompts = [
        prompt,
        "Sorry, I did not catch that clearly. Could you say just your first name?"
    ]

    for p in prompts:
        raw_name = robot_listen(p, pepper).strip()

        if "interrupt" in raw_name.lower():
            raise KeyboardInterrupt("User interrupted during name input.")

        name = clean_spoken_name(raw_name)

        if name != "Guest":
            return name

    return "Guest"

def ask_and_confirm_name(prompt: str, pepper=None) -> str:
    """
    Ask for a name and confirm it once.
    Falls back to Guest if the name is unclear.
    """
    for _ in range(2):
        raw_name = robot_listen(prompt, pepper).strip()

        if "interrupt" in raw_name.lower():
            raise KeyboardInterrupt("User interrupted during name input.")

        name = clean_spoken_name(raw_name)

        if name == "Guest":
            prompt = "I did not catch the name clearly. Could you say just your name?"
            continue

        confirm = robot_listen(
            f"I heard {name}. Is that correct?",
            pepper
        ).strip()

        if "interrupt" in confirm.lower():
            raise KeyboardInterrupt("User interrupted during name confirmation.")

        if _is_yes(confirm):
            return name

        if _is_no(confirm):
            prompt = "Sorry about that. What name should I use?"
            continue

    return "Guest"

#  REGISTRATION
def register_new_user(encoding: np.ndarray, db: dict, pepper=None) -> str:
    """
    Temporary new-user flow.

    Face recognition does not save anything here.
    It only asks what to call the person during this interaction and stores
    the face encoding temporarily so main.py can ask for save consent later.
    """
    global _PENDING_NEW_USER_ENCODING
    _PENDING_NEW_USER_ENCODING = encoding

    robot_act_and_say(
        robot_wave,
        "Hello, I am SocioBot. I help Homo sapiens choose events based on their mood, budget, group size, and preferences.",
        pepper,
        speech_delay=0.2,
    )

    name = ask_temporary_name(
        "What should I call you?",
        pepper
    )

    robot_say(f"Nice to meet you, {name}.", pepper)

    print(f"[INFO] New user detected as temporary user '{name}'. Face not saved yet.")
    return name


#  MAIN RECOGNITION LOOP
def run_face_recognition(pepper=None) -> tuple[str, str]:
    """
    Open webcam, detect and identify the first confirmed face.

    Returns
    -------
    (status, name)
      "known"   -> returning user (greeted by name)
      "new"     -> just registered
      "unknown" -> user quit camera before recognition
    """
    db  = load_known_faces()
    cap = cv2.VideoCapture(CAP_V4L2)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check camera connection.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    for _ in range(15):
        ret, _ = cap.read()
        if not ret:
            time.sleep(0.05)



    confirm_buffer  = {}
    unknown_counter = 0
    no_face_frames  = 0
    last_enc        = None
    result          = None

    last_idle_motion = 0.0

    print("[INFO] Face recognition started. Press U to stop SocioBot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame, retrying...")
            time.sleep(0.05)
            continue

        # Guard against black/corrupted frames
        if frame is None or frame.mean() < 1.0:
            print("[WARN] Received black/empty frame, skipping.")
            continue


        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(frame, (640, 480))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        # small = cv2.resize(frame, (640, 480), fx=0.5, fy=0.5)
        # rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = fr.face_locations(rgb, model="hog")
        encodings = fr.face_encodings(rgb, locations)

        # Scale bounding boxes back to original resolution
        # locations_full = [(t*2, r*2, b*2, l*2) for t, r, b, l in locations]

        scale_x = frame.shape[1] / 640
        scale_y = frame.shape[0] / 480
        locations_full = [
            (int(t * scale_y), int(r * scale_x), int(b * scale_y), int(l * scale_x))
            for t, r, b, l in locations
        ]

        if not encodings:
            if quick_face_present(gray):
                no_face_frames = 0  # Haar still sees something - transient glitch
            else:
                no_face_frames += 1
                if no_face_frames >= NO_FACE_GRACE:
                    # Person has genuinely left - full reset
                    unknown_counter = 0
                    confirm_buffer  = {}
                    last_enc        = None
                    no_face_frames  = 0

            cv2.putText(
                frame, f"Looking for face... ({no_face_frames}/{NO_FACE_GRACE})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1
            )
            if time.time() - last_idle_motion >= IDLE_PATROL_INTERVAL_SEC:
                robot_idle_patrol(pepper)
                last_idle_motion = time.time()

        else:
            no_face_frames = 0  # face is back
            robot_stop_motion(pepper)

            for enc, (top, right, bottom, left) in zip(encodings, locations_full):
                last_enc     = enc
                matched_name = identify_face(enc, db)

                if matched_name:
                    # Known user
                    unknown_counter = 0
                    confirm_buffer[matched_name] = confirm_buffer.get(matched_name, 0) + 1

                    color = (0, 200, 0)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(
                        frame,
                        f"{matched_name} ({confirm_buffer[matched_name]}/{CONFIRMATION_FRAMES})",
                        (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                    )

                    if confirm_buffer[matched_name] >= CONFIRMATION_FRAMES:
                        cap.release()
                        cv2.destroyAllWindows()
                        robot_stop_motion(pepper)
                        return ("known", matched_name)

                else:
                    # Unknown user
                    unknown_counter += 1
                    confirm_buffer   = {}

                    color = (0, 0, 220)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(
                        frame,
                        f"Unknown ({unknown_counter}/{UNKNOWN_PATIENCE})",
                        (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                    )

                    if unknown_counter >= UNKNOWN_PATIENCE:
                        cap.release()
                        cv2.destroyAllWindows()
                        enc_to_use = last_enc if last_enc is not None else enc
                        name = register_new_user(enc_to_use, db, pepper)
                        return ("new", name)

        cv2.imshow("SocioBot - Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("u"), ord("U")):
            cap.release()
            cv2.destroyAllWindows()
            robot_stop_motion(pepper)
            return ("manual_stop", "unidentified")

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return ("unknown", "unidentified")


#  UTILITIES
def list_registered_users() -> list[str]:
    return DEFAULT_FACE_DATABASE.list_registered_users()


def delete_user(name: str) -> bool:
    return DEFAULT_FACE_DATABASE.delete_user(name)


class NewUserRegistration:
    """Coordinates temporary user naming and deferred face-data consent."""

    def __init__(self, pepper=None, database: Optional[FaceDatabase] = None):
        self.pepper = pepper
        self.database = database or DEFAULT_FACE_DATABASE

    def ask_storage_consent(self) -> bool:
        return ask_face_storage_consent(self.pepper)

    def clean_name(self, raw_name: str) -> str:
        return clean_spoken_name(raw_name)

    def ask_temporary_name(self, prompt: str) -> str:
        return ask_temporary_name(prompt, self.pepper)

    def ask_and_confirm_name(self, prompt: str) -> str:
        return ask_and_confirm_name(prompt, self.pepper)

    def register(self, encoding: np.ndarray, db: dict) -> str:
        return register_new_user(encoding, db, self.pepper)

    def has_pending_encoding(self) -> bool:
        return has_pending_new_user_encoding()

    def save_pending(self, name: str, interest: str = "Unknown") -> bool:
        return save_pending_new_user(name, interest)

    def discard_pending(self) -> None:
        discard_pending_new_user_encoding()


class FaceRecognizer:
    """Object-oriented entry point for face identification and webcam recognition."""

    def __init__(self, pepper=None, database: Optional[FaceDatabase] = None):
        self.pepper = pepper
        self.database = database or DEFAULT_FACE_DATABASE
        self.registration = NewUserRegistration(pepper, self.database)

    def identify(self, encoding: np.ndarray, db: Optional[dict] = None) -> Optional[str]:
        active_db = self.database.load() if db is None else db
        return identify_face(encoding, active_db)

    def quick_face_present(self, gray_frame: np.ndarray) -> bool:
        return quick_face_present(gray_frame)

    def run(self) -> tuple[str, str]:
        return run_face_recognition(self.pepper)


# Standalone test
if __name__ == "__main__":
    print("Running face recognition standalone (no robot)...")
    status, name = run_face_recognition(pepper=None)
    print(f"\nResult --> status={status!r}, name={name!r}")
