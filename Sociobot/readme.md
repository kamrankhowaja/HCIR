# SocioBot

**From mood to move — in seconds**

SocioBot is a social event recommendation robot. It talks with a user, recognizes whether they are a returning or new visitor, asks a few simple questions about their mood, budget, group size, weather, and preferences, then recommends an event that fits their situation.

The project can run with the qiBullet Pepper robot simulation, or in text-only mode if you do not want to launch the robot simulator.

## What SocioBot Does

- Recognizes known and new users using face recognition.
- Collects event preferences through a short dialog.
- Uses a Bayesian network to reason about the best event match.
- Recommends events such as concerts, food festivals, sports events, museum visits, student meetups, board game nights, and picnics.
- Saves returning-user preferences when the user gives permission.

## Project Files

- `main.py` starts the SocioBot application.
- `modules/` contains the robot interaction, dialog, face recognition, and recommendation logic.
- `requirements.txt` lists the Python dependencies.
- `known_faces/` stores saved face data.
- `demo_output/` contains Bayesian network demo output files.

## Setup

These steps create a Python virtual environment, install the dependencies, and run SocioBot.

### 1. Create the venv

```bash
python3 -m venv venv
```

### 2. Activate the venv

On Linux or macOS:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Project

To run SocioBot with the qiBullet robot simulation:

```bash
python main.py
```

To run SocioBot in text-only mode:

```bash
python main.py --no-robot
```

## Stop the Program

Press `U`  in the terminal or say `Interrupt` while robot is listening to stop SocioBot.

