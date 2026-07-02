# Technical Documentation & Integration Guide
**Module:** Face Recognition, Voice Activation & Session Management  
**Robot Agent Identity:** `EventMate`

---

### Requirements

##### A. Linux / Ubuntu Audio Dependencies (Fixes the ALSA/Jack Bugs)
Because PyAudio needs to communicate directly with your computer's sound cards, you must install the native development headers on your system before running pip install:

``` bash
Bash

# Update your system package repository
sudo apt-get update

# Install portaudio development headers (Absolute must-have for PyAudio)
sudo apt-get install -y portaudio19-dev python3-pyaudio

# Optional: Mute the ALSA driver spam configuration errors in the terminal
sudo apt-get install -y alsa-utils
```


##### B Compilation Tools for face_recognition
The face_recognition package relies on a heavy C++ toolkit called dlib. If pip fails or hangs while installing it, it means your operating system is missing a C++ compiler or the build configuration utility cmake. Fix this by running:

```bash
Bash
sudo apt-get install -y build-essential cmake
```

##### C. qiBullet Environment Setup

##### D. Verification Commands for Team

Once everything is ready, run the initialization commands in this order inside their terminal:

```bash
# 1. Install system tools first
sudo apt-get install -y portaudio19-dev build-essential cmake

# 2. Install Python packages from requirements
pip install -r face_recognition_module_requirements.txt

# 3. Double check qiBullet is attached
pip install qibullet
```





## 1. Module Overview
This module acts as the entry point and gatekeeper for the entire robot application. It continuously coordinates two execution pipelines in a non-blocking configuration:
1. **The Core Visual Thread (Main Thread):** Renders the live OpenCV webcam window and tracks face presence/biometrics without freezing.
2. **The Interaction Subsystem (Background Worker):** Handles proactive voice wake-words (`"hello"`, `"need help"`), Text-to-Speech (TTS), Speech-to-Text (STT), a privacy consent gateway, and a 10-second user presence heartbeat.

---

## 2. High-Level Behavioral Flow Matrix

1. **Idle Monitoring:** Robot remains stationary. It scans for a face or background voice triggers (`"hello"`, `"I need help"`).
2. **Proactive Activation:** If a face is tracked consistently for >= 2 seconds OR a wake-word is heard, `EventMate` waves and gives an identity greeting.
3. **Privacy Consent Gate:** - **New Users:** Prompted for their name (speech cleaned automatically). They must explicitly verbally consent to data tracking. If **Yes**, features are saved to local database dictionary (`face_data.pkl`). If **No**, the session runs strictly as an **Anonymous Guest** in temporary RAM.
   - **Known Users:** Directly identified via biometrics, bypassing onboarding checks.
4. **Presence Heartbeat Check:** Every 10 seconds, if the camera completely loses the user's face, the background thread halts the dialogue to ask *"Are you still there?"*. It actively scans and listens for a 5-second grace window. If a verbal response or face returns, it resumes; otherwise, it cleanly terminates and stands by for the next visitor.

---

## 3. The Shared Integration Layer (`RobotSessionContext`)

To prevent multi-threading race conditions or resource locking, all communication states are exposed via a thread-safe global object: `session_context`. 

**You must import this object to monitor or control the session state.**

### State Properties to Read:
| Variable Name | Type | Allowed Values | Target Module & Purpose |
| :--- | :--- | :--- | :--- |
| `session_context.session_active` | `bool` | `True` / `False` | **Dialog Tree / Bayesian Net:** Wrap your main loops in `while session_context.session_active:` to ensure you only execute when a user is present. |
| `session_context.user_name` | `str` | Clean Name (e.g., `"Rahul"`) or `"Guest_123"` | **Dialog Tree:** Use this string dynamically to personalize your speech outputs. (e.g., *"Welcome Rahul, what events do you like?"*) |
| `session_context.status` | `str` | `"known"` / `"new"` / `"anonymous"` | **Bayesian Network:** <br>• If `"known"`, load their historical preferences directly.<br>• If `"new"` or `"anonymous"`, prompt the initial recommendation survey questions. |
| `session_context.is_consent_granted`| `bool` | `True` / `False` | **Data Storage:** Confirms our modules are permitted to save preference profiles to long-term storage or tracking files. |

---

## 4. How we can Integrate Their Code (Step-by-Step)

### Step A: Importing and Hooking into the Dialogue Loop
Inside the face recognition file, look for the internal pipeline method named `_mock_handoff_loop()`. This is exactly where the dialogue and recommendation modules should insert their entry point functions.

**Inside your module (`face_recognition_module_with_Robot.py`):**
```python
def _mock_handoff_loop(self):
    """
    Simulates interaction runtime execution, holding status parameters open 
    for other module hooks.
    """
    print(f"[PIPELINE RUNNING] Handed control to core application modules.")
    
    # ── Other module INSERT THEIR CODE HOOK HERE ──
    # Example:
    # import dialog_engine
    # dialog_engine.run_event_recommendation_flow(session_context)
    
    while session_context.session_active:
        time.sleep(0.5)

```


#### Step B: Writing the Teammate's Dialogue Loop Safely
When writing the `Dialog Module or Bayesian Engine`, we must make sure the other script monitors my thread states and handles user termination correctly.

Here is a clean code skeleton of how their separate file (`dialog_engine.py`) should look:
```python

# dialog_engine.py
import time

def run_event_recommendation_flow(session_context):
    """
    This function is executed inside EventMate's active interaction thread.
    """
    # 1. Check user status to guide the Bayesian Network behavior
    if session_context.status == "known":
        print(f"Loading pre-existing profile weights for {session_context.user_name}...")
        # Skip survey logic, fetch known preferences from database
    else:
        print(f"Initializing a clean Bayesian context graph for {session_context.user_name}...")
        # Prepare to ask onboarding event type questions
        
    # 2. Main Dialogue loop wrapped cleanly around your session flag
    while session_context.session_active:
        
        # Example processing task:
        # user_input = robot_listen_blocking("What kind of music do you like?")
        
        # ... Run Bayesian inference and give responses ...
        
        # 3. CRITICAL: If the user says goodbye or wants to quit,
        # the dialog system MUST shut down the active session context flag!
        if "goodbye" in user_input or "exit" in user_input:
            print("[DIALOG ENGINE] User requested termination.")
            
            # Setting this to False tells the face recognition script 
            # to say goodbye, trigger default postures, and reset for the next user.
            session_context.session_active = False 
            break
            
        time.sleep(0.1)


```




## 5. Summary Checkpoints for the Team

* **Microphone Access Control:** Do not make raw calls to `speech_recognition` directly inside subsequent modules. Use the non-blocking framework helpers (`robot_say_blocking("text")` and `robot_listen_blocking("prompt")`) to keep the robot and audio systems safe from multi-threaded hardware crashes.
* **Session Interruption Handling:** If the user steps away from the camera for more than 10 seconds, `session_active` will flip to `False` from the background. The dialogue loop will drop immediately, protecting the code from getting trapped waiting for answers in an empty room.