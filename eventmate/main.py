"""
main.py
-------
EventMate — Social Event Recommendation Robot

Interaction flow:
  1. Launch qiBullet simulation (Pepper robot)
  2. Face recognition - identify user (known / new / unknown)
  3. Build initial BN evidence from recognition result
  4. Dialog - collect 4 preference answers
  5. Bayesian Network inference - ranked event recommendations
  6. Present top-3 recommendations with gestures
  7. Save user's interest for next session
  8. Farewell sequence - reset for next user

Run:
  python main.py
  python main.py --no-robot     (skip qiBullet, text-only mode)
  python main.py --no-tts       (skip text-to-speech)
"""

from __future__ import annotations
import sys
import time
import argparse

# Module imports
from modules.gestures import (
    launch_simulation, robot_act_and_say,
    robot_say, robot_nod,
    robot_wave, robot_thinking,
    robot_thumbs_up, robot_excited,
    robot_farewell, robot_point
)
from modules.face_recognition import (
    run_face_recognition,
    get_preference_history,
    save_user_preference
)
from modules.dialog_manager import DialogManager
from modules.event_recommendation_bn import recommend_events


#  RECOMMENDATION PRESENTER

# Human-readable explanations for each event
EVENT_DESCRIPTIONS = {
    "Concert":       "a live music concert — great for music lovers in an energetic crowd",
    "FoodFestival":  "a food festival — perfect for foodies who enjoy variety and outdoor fun",
    "SportsEvent":   "a sports event — ideal for active, competitive, group-oriented people",
    "MuseumVisit":   "a museum visit — a calm cultural experience, great for solo or small groups",
    "StudentMeetup": "a student meetup — excellent for networking and meeting new people",
    "BoardGameNight":"a board game night — fun, relaxed, and perfect for small groups on a budget",
    "Picnic":        "a picnic — a low-cost, outdoor, easygoing option for any group size",
}


def present_recommendations(
    ranking: list,
    inferred: dict,
    pepper=None
) -> str:
    """
    Present top-3 recommendations with gestures and speech.
    Returns the top recommendation name.
    """
    top_event, top_prob = ranking[0]
    description = EVENT_DESCRIPTIONS.get(top_event, top_event)

    # Top pick
    robot_act_and_say(
        robot_thumbs_up,
        f"My top recommendation for you is {description}!",
        pepper,
        speech_delay=0.2,
    )
    time.sleep(0.5)

    # Reasoning (explain from inferred nodes)
    top_interest = max(inferred["InterestType"], key=inferred["InterestType"].get)
    top_score    = max(inferred["EventScoreMatching"], key=inferred["EventScoreMatching"].get)

    robot_say(
        f"Based on your preferences, I believe your main interest leans toward "
        f"{top_interest.lower()}, and your event match score is {top_score.lower()}.",
        pepper
    )
    time.sleep(0.3)

    # Runner-up suggestions
    if len(ranking) >= 2:
        second_event = ranking[1][0]
        third_event  = ranking[2][0] if len(ranking) >= 3 else None

        if third_event:
            robot_act_and_say(
                robot_point,
                f"Other good options are: {second_event} and {third_event}.",
                pepper,
                speech_delay=0.15,
            )
        else:
            robot_act_and_say(
                robot_point,
                f"Another good option would be {second_event}.",
                pepper,
                speech_delay=0.15,
            )

    # Print full ranking to terminal
    print("\n" + "═" * 50)
    print("  EVENT RECOMMENDATIONS")
    print("═" * 50)
    for rank, (event, prob) in enumerate(ranking, 1):
        bar  = "█" * int(prob * 35)
        star = " *" if rank == 1 else ""
        print(f"  {rank}. {event:<20} {prob:.3f}  {bar}{star}")
    print("═" * 50)

    print("\n  INFERRED CONTEXT")
    for node, dist in inferred.items():
        top = max(dist, key=dist.get)
        print(f"  {node:<24} - {top}  (p={dist[top]:.3f})")
    print("═" * 50 + "\n")

    return top_event


#  FAREWELL SEQUENCE
def farewell_sequence(name: str, top_event: str, pepper=None) -> None:
    """Personalised farewell with gesture."""

    if name and name not in ("unidentified", "Guest"):
        msg = (
            f"I hope you have a wonderful time at the {top_event}, {name}! "
            f"Come back and see me anytime."
        )
    else:
        msg = (
            f"I hope you enjoy the {top_event}! "
            f"Have a great time and come back soon!"
        )

    robot_act_and_say(
        robot_farewell,
        msg,
        pepper,
        speech_delay=0.8,
    )
    time.sleep(2.0)


#  MAIN INTERACTION LOOP
def run_interaction(pepper=None) -> None:
    """Execute one full user interaction from recognition to farewell."""

    # Face recognition
    print("\n[STEP 1] Running face recognition...")
    status, name = run_face_recognition(pepper)

    print(f"\n Recognition result: status={status!r}, name={name!r}")

    if status == "unknown":
        robot_say(
            "I'm sorry, I was unable to identify you. "
            "Please try again when you're ready!",
            pepper
        )
        return

    # Build initial evidence from face recognition
    print("\n[STEP 2] Building initial evidence...")
    evidence: dict[str, str] = {}

    if status == "known":
        evidence["RecognizedUser"] = "Known"
        history = get_preference_history(name)
        evidence["UserPreferenceHistory"] = history
        print(f"  - Known user. History preference: {history}")

        # Extra enthusiasm for returning users
        robot_act_and_say(
            robot_excited,
            f"Wonderful to have you back, {name}! "
            f"Let me help you find something great today.",
            pepper,
            speech_delay=0.2,
        )

    elif status == "new":
        evidence["RecognizedUser"]        = "Unknown"
        evidence["UserPreferenceHistory"] = "Unknown"
        print("  - New user. No preference history available.")
        robot_act_and_say(
            robot_wave,
            f"Welcome, {name}! I'm EventMate, your social event assistant. "
            f"Let me get to know your preferences.",
            pepper,
            speech_delay=0.25,
        )

    time.sleep(0.5)

    # Dialog — collect preferences
    print("\n[STEP 3] Running preference dialog...")
    dm = DialogManager(pepper=pepper)
    preference_evidence = dm.run()
    evidence.update(preference_evidence)

    print("\n  - Final evidence dict:")
    for node, value in evidence.items():
        print(f"     {node:<26} = {value}")

    # Bayesian Network inference
    print("\n[STEP 4] Running Bayesian inference...")
    robot_thinking(pepper, duration=2.5)

    try:
        ranking, inferred = recommend_events(evidence)
        print("  - Inference complete.")
    except Exception as e:
        print(f"  ✗ Inference error: {e}")
        robot_say(
            "I'm sorry, something went wrong with my reasoning engine. "
            "Please try again!",
            pepper
        )
        return

    # Present recommendations
    print("\n[STEP 5] Presenting recommendations...")
    top_event = present_recommendations(ranking, inferred, pepper)

    # Save user interest for personalisation
    if status == "known" and name not in ("unidentified", "Guest"):
        top_interest = max(
            inferred["InterestType"], key=inferred["InterestType"].get
        )
        save_user_preference(name, top_interest)
        print(f" Saved interest '{top_interest}' for user '{name}'.")

    # Farewell
    print("\n[STEP 6] Farewell sequence...")
    farewell_sequence(name, top_event, pepper)
    print("\n Interaction complete. Resetting...\n")


#  ENTRY POINT
def parse_args():
    parser = argparse.ArgumentParser(description="EventMate — Social Event Recommendation Robot")
    parser.add_argument(
        "--no-robot", action="store_true",
        help="Run without qiBullet simulation (text-only mode)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║       EventMate — CP2 Demo               ║")
    print("║  Social Event Recommendation Robot       ║")
    print("╚══════════════════════════════════════════╝\n")

    # Launch robot simulation
    pepper      = None
    sim_manager = None

    if not args.no_robot:
        print("[INIT] Launching qiBullet simulation...")
        pepper, sim_manager = launch_simulation()
    else:
        print("[INIT] --no-robot flag set. Running in text-only mode.")

    # Main loop — keep running until keyboard interrupt
    try:
        while True:
            run_interaction(pepper)

            again = input(
                "\nAnother interaction? (y/n): "
            ).strip().lower()

            if again != "y":
                break

    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user.")

    finally:
        print("\n[SHUTDOWN] Cleaning up...")
        if sim_manager is not None:
            try:
                sim_manager.stopSimulation()
                print("[SHUTDOWN] qiBullet stopped.")
            except Exception as e:
                print(f"[WARN] Simulation stop error: {e}")
        print("[SHUTDOWN] EventMate offline. Goodbye!")


if __name__ == "__main__":
    main()
