"""
Face Detection and Recognition module for EventMate.

This module handles ONLY the vision logic:
  - Loading / saving the face database
  - Identifying a face encoding against known users
  - Running the live webcam recognition loop

Run standalone to test:
  python modules/face_recognition.py
"""

from __future__ import annotations
import importlib
import os
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
    from modules.gestures import robot_act_and_say, robot_say, robot_wave, robot_nod, robot_listen
except ModuleNotFoundError:
    from gestures import robot_act_and_say, robot_say, robot_wave, robot_nod, robot_listen

# Config
DATA_DIR            = "./known_faces"
DATA_FILE           = os.path.join(DATA_DIR, "face_data.pkl")

TOLERANCE           = 0.50   # lower = stricter (0.4–0.6 typical)
CONFIRMATION_FRAMES = 5      # consecutive matched frames to confirm identity
UNKNOWN_PATIENCE    = 15     # unknown frames before triggering registration
NO_FACE_GRACE       = 15     # no-face frames before resetting counters

# Haar cascade for fast pre-screening
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)


#  PERSISTENCE
def load_known_faces() -> dict:
    """Load {name: [encodings]} from disk. Returns empty dict if not found."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_known_faces(db: dict) -> None:
    """Persist face database to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DATA_FILE, "wb") as f:
        pickle.dump(db, f)


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
    Fast Haar cascade check — True if any face visible.
    Used as a grace-period guard so a single missed frame doesn't reset counters.
    """
    faces = _face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
    )
    return len(faces) > 0


def get_preference_history(name: str) -> str:
    """
    Return the most recent interest category stored for a known user,
    or 'Unknown' if the user has no stored preference.
    """
    db = load_known_faces()
    entry = db.get(name, None)

    if entry is None:
        return "Unknown"

    if isinstance(entry, dict):
        return entry.get("last_interest", "Unknown")

    return "Unknown"


def save_user_preference(name: str, interest: str) -> None:
    """Persist the user's last interest category after a session."""
    db = load_known_faces()
    entry = db.get(name, None)

    if entry is None:
        return  # Unknown user, nothing to update

    if isinstance(entry, list):
        # Upgrade legacy format
        encodings = entry
        db[name] = {"encodings": encodings, "last_interest": interest}
    elif isinstance(entry, dict):
        entry["last_interest"] = interest

    save_known_faces(db)
    print(f"[INFO] Saved preference '{interest}' for user '{name}'.")


#  REGISTRATION
def register_new_user(encoding: np.ndarray, db: dict, pepper=None) -> str:
    """
    Spoken registration flow.
    Asks for the user's name via TTS + STT (or keyboard fallback),
    saves encoding, and returns the name.
    """
    robot_say("Hello there! I have not seen you before.", pepper)
    time.sleep(0.3)

    name = robot_listen("What is your name, please?", pepper)
    if not name:
        name = "Guest"

    robot_act_and_say(
        robot_wave,
        f"Nice to meet you, {name}! I will remember your face for next time.",
        pepper,
        speech_delay=0.2,
    )

    # Save in dict format for forward compatibility
    if name not in db:
        db[name] = {"encodings": [], "last_interest": "Unknown"}

    entry = db[name]
    if isinstance(entry, list):
        entry.append(encoding)
    else:
        entry["encodings"].append(encoding)

    save_known_faces(db)
    print(f"[INFO] User '{name}' registered.")
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
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check camera connection.")

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
        small = cv2.resize(frame, (640, 480), fx=0.5, fy=0.5)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = fr.face_locations(rgb, model="hog")
        encodings = fr.face_encodings(rgb, locations)

        # Scale bounding boxes back to original resolution
        locations_full = [(t*2, r*2, b*2, l*2) for t, r, b, l in locations]

        if not encodings:
            if quick_face_present(gray):
                no_face_frames = 0  # Haar still sees something — transient glitch
            else:
                no_face_frames += 1
                if no_face_frames >= NO_FACE_GRACE:
                    # Person has genuinely left — full reset
                    unknown_counter = 0
                    confirm_buffer  = {}
                    last_enc        = None
                    no_face_frames  = 0

            cv2.putText(
                frame, f"Looking for face... ({no_face_frames}/{NO_FACE_GRACE})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1
            )

        else:
            no_face_frames = 0  # face is back

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
                        robot_act_and_say(
                            robot_wave,
                            f"Welcome back, {matched_name}! Great to see you again.",
                            pepper,
                            speech_delay=0.2,
                        )
                        robot_nod(pepper)
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

        cv2.imshow("EventMate — Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return ("unknown", "unidentified")


#  UTILITIES
def list_registered_users() -> list[str]:
    return list(load_known_faces().keys())


def delete_user(name: str) -> bool:
    db = load_known_faces()
    if name in db:
        del db[name]
        save_known_faces(db)
        print(f"[INFO] User '{name}' deleted.")
        return True
    print(f"[INFO] User '{name}' not found.")
    return False


# Standalone test
if __name__ == "__main__":
    print("Running face recognition standalone (no robot)...")
    status, name = run_face_recognition(pepper=None)
    print(f"\nResult → status={status!r}, name={name!r}")
