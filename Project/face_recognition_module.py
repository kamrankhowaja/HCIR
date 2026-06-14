
"""
face_recognition_module.py
----------------------------
Face Recognition Module for the Preference-Based Social Event Recommendation Robot.

Behaviour:
  - Robot watches live webcam feed continuously.
  - If a face is detected:
      * KNOWN  --> greets user by name, returns ("known", name)
      * UNKNOWN --> prompts user to enter their name, saves face encoding to disk,
                  returns ("new", name)

Storage (no database required):
  - known_faces/face_data.pkl  -->  dict { name: [list of 128-d encodings] }

Dependencies:
  pip install face_recognition opencv-python numpy
"""
from __future__ import annotations
from typing import Optional
import os
import cv2
import pickle
import numpy as np
# pyrefly: ignore [missing-import]
import face_recognition


# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR   = "known_faces"
DATA_FILE  = os.path.join(DATA_DIR, "face_data.pkl")

TOLERANCE           = 0.50   # lower = stricter match (0.4-0.6 is typical)
CONFIRMATION_FRAMES = 5     # consecutive matched frames before confirming known identity

# Unknown face stability settings
UNKNOWN_PATIENCE    = 15     # total unknown frames needed to trigger registration
NO_FACE_GRACE       = 15     # frames WITHOUT a face allowed before resetting unknown counter
                             # (prevents a single missed frame from resetting the count)
# ────────────────────────────────────────────────────────────────────────────


# ── Haar Cascade face detector (OpenCV) for fast pre-screening ───────────────
# Used to quickly check if ANY face is visible before running the heavier
# face_recognition encoding step.
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)



# ── Persistence helpers ──────────────────────────────────────────────────────

def load_known_faces() -> dict:
    """Load saved face encodings from disk. Returns {name: [encodings]}."""
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


# ── Core recognition helpers ─────────────────────────────────────────────────

def identify_face(encoding: np.ndarray, db: dict) -> Optional[str]:
    """
    Compare an encoding against every stored user.
    Returns the best-matching name (str) or None if no match found.
    """
    best_name = None
    best_dist = float("inf")

    for name, stored_encodings in db.items():
        distances = face_recognition.face_distance(stored_encodings, encoding)
        min_dist  = float(np.min(distances))
        if min_dist < best_dist:
            best_dist = min_dist
            best_name = name

    if best_dist <= TOLERANCE:
        return best_name
    return None


def register_new_user(encoding: np.ndarray, db: dict) -> str:
    """
    Ask the robot (or a fallback terminal prompt) for the user's name,
    then store their encoding and persist to disk.

    In a qiBullet/Pepper integration replace input() with the robot's
    text-to-speech + speech-to-text pipeline.
    """
    # ── Replace this block with Pepper TTS/STT if running on the robot ──
    print("\n[ROBOT] Hi! I don't recognise you yet.")
    name = input("[ROBOT] What is your name? ").strip()
    if not name:
        name = "Guest"
    # ────────────────────────────────────────────────────────────────────

    if name not in db:
        db[name] = []
    db[name].append(encoding)
    save_known_faces(db)

    print("[ROBOT] Nice to meet you, {}! I'll remember you next time.".format(name))
    return name


def quick_face_present(gray_frame: np.ndarray) -> bool:
    """
    Fast OpenCV Haar cascade check — returns True if at least one face
    is visible in the frame. Used to gate the grace period: we only
    let the unknown_counter keep its value when NO face is detected
    briefly, but we do NOT reset it just because Haar missed for a frame.
    """
    faces = _face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(60, 60)
    )
    return len(faces) > 0


# ── Main recognition loop ────────────────────────────────────────────────────

def run_face_recognition() -> tuple[str, str]:
    """
    Open the webcam, detect faces, identify or register the user.

    Key stability improvement:
      - unknown_counter only resets after NO_FACE_GRACE consecutive frames
        with no face detected at all (via Haar cascade). A single missed
        frame no longer resets progress.

    Returns
    -------
    (status, name)
      status : "known" | "new" | "unknown"
      name   : str
    """
    db  = load_known_faces()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check camera connection.")

    confirm_buffer  = {}   # name -> consecutive matched frame count
    unknown_counter = 0    # total frames face was seen but unrecognised
    no_face_frames  = 0    # consecutive frames with NO face visible at all
    last_enc        = None # last valid encoding (used to hold state across missed frames)
    result          = None

    print("[INFO] Starting face recognition. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # ── Detect with face_recognition (accurate, slower) ──────────────
        locations = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, locations)
        locations_full = [(t*2, r*2, b*2, l*2) for t, r, b, l in locations]

        face_seen_this_frame = len(encodings) > 0

        if not face_seen_this_frame:
            # ── No face from face_recognition — check with faster Haar ───
            haar_sees_face = quick_face_present(gray)

            if haar_sees_face:
                # Haar still sees something — likely a pose/lighting glitch
                # Keep counters as-is (grace: do nothing this frame)
                no_face_frames = 0
            else:
                # Genuinely no face in frame
                no_face_frames += 1
                if no_face_frames >= NO_FACE_GRACE:
                    # Person has actually left — full reset
                    unknown_counter = 0
                    confirm_buffer  = {}
                    last_enc        = None
                    no_face_frames  = 0

            # Draw current status on frame
            status_text = "No face detected ({}/{} to reset)".format(no_face_frames, NO_FACE_GRACE)
            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        else:
            no_face_frames = 0  # reset grace counter — face is back

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
                        print("\n[ROBOT] Welcome back, {}!".format(matched_name))
                        result = ("known", matched_name)
                        break

                else:
                    # ── Unknown user — increment stably ───────────────────
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
                        name = register_new_user(enc_to_save, db)
                        return ("new", name)

        cv2.imshow("Robot - Face Recognition", frame)

        if result or (cv2.waitKey(1) & 0xFF == ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()

    if result:
        return result
    return ("unknown", "unidentified")


# ── Integration interface (used by main robot pipeline) ─────────────────────

def get_user_status() -> tuple[str, str]:
    """
    Entry point called by the robot's main loop or state machine.

    Returns
    -------
    (status, name)
      status = "known"   -> returning user, proceed directly to recommendation
      status = "new"     -> just registered; run preference survey first
      status = "unknown" -> camera closed before recognition (handle gracefully)
    """
    return run_face_recognition()


# ── Utility: list / delete registered users ──────────────────────────────────

def list_registered_users() -> list[str]:
    """Return names of all stored users."""
    db = load_known_faces()
    return list(db.keys())


def delete_user(name: str) -> bool:
    """Remove a user and all their encodings from the database."""
    db = load_known_faces()
    if name in db:
        del db[name]
        save_known_faces(db)
        print("[INFO] User '{}' deleted.".format(name))
        return True
    print("[INFO] User '{}' not found.".format(name))
    return False


# ── Standalone entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    status, name = get_user_status()
    print("\nResult -> status={!r}, name={!r}".format(status, name))