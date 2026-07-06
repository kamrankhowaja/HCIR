"""
SocioBot  — Social Event Recommendation Robot
"""

from __future__ import annotations
import re
import time
import random
import argparse

# Module imports
from modules.gestures import (
    RobotInteractionLayer,
    launch_simulation, robot_act_and_say,
    robot_say, robot_listen,
    robot_wave,
    robot_present,
    robot_farewell, robot_point
)
from modules.face_recognition import (
    FaceRecognizer,
    get_preference_history,
    save_user_preference,
    save_pending_new_user,
    discard_pending_new_user_encoding,
    has_pending_new_user_encoding
)
from modules.dialog_manager import DialogManager
from modules.event_recommendation_bn import EventRecommendationNetwork


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

def _normalize_choice_answer(answer: str) -> str:
    return re.sub(r"[^a-zA-Z'\s]", " ", answer.lower()).strip()


def _contains_choice_phrase(text: str, phrases: list[str]) -> bool:
    text = f" {text} "
    return any(f" {phrase} " in text for phrase in phrases)


def _is_yes_response(answer: str) -> bool:
    a = _normalize_choice_answer(answer)
    return _contains_choice_phrase(a, [
        "yes", "yeah", "yep", "sure", "okay",
        "i would", "please do", "go ahead",
        "sounds good", "yes please"
    ])


def _is_no_response(answer: str) -> bool:
    a = _normalize_choice_answer(answer)
    return _contains_choice_phrase(a, [
        "no", "nope", "not now", "not today",
        "later", "stop", "cancel", "don't", "dont"
    ])

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
        robot_present,
        f"My top recommendation for you is {description}!",
        pepper,
        speech_delay=0.2,
    )
    time.sleep(0.5)

    # Reasoning (explain from inferred nodes)
    top_interest = max(inferred["InterestType"], key=inferred["InterestType"].get)
    top_score    = max(inferred["EventScoreMatching"], key=inferred["EventScoreMatching"].get)

    robot_say(
        f"From what you told me, I think you would enjoy this based on your mood, "
        f"budget, and plans today.",
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
                f"If you want alternatives, I would also keep {second_event} and {third_event} in mind.",
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

def introduce_sociobot(status: str, name: str, pepper=None) -> None:
    """
    Introduce SocioBot after a known face has been detected.
    """
    intro = random.choice([
        f"Hello {name}. I am SocioBot. I help Homo sapiens choose events based on their mood, budget, group size, and preferences. From mood to move, in seconds.",
        f"Hi {name}, welcome back. I am SocioBot. Tell me a little about your plans, and I can suggest something that fits. From mood to move — in seconds.",
        f"Hey {name}. SocioBot is back online. I can help you find something interesting to do today. From mood to move — in seconds."
    ])

    robot_act_and_say(
        robot_wave,
        intro,
        pepper,
        speech_delay=0.2,
    )

def ask_recommendation_consent(status: str, name: str, pepper=None) -> bool:
    """
    Ask naturally whether the user wants an event recommendation.
    Returns True if the dialog should continue.
    """
    if status == "known" and name not in ("unidentified", "Guest"):
        prompt = random.choice([
            f"Would you like me to recommend something for today, {name}?",
            f"Should I suggest an event that fits your plans today, {name}?",
            f"Would you like a personalized event recommendation now, {name}?"
        ])
    else:
        prompt = random.choice([
            "Would you like me to recommend an event for today?",
            "Should I suggest something interesting to do today?",
            "Would you like a few event ideas based on your preferences?"
        ])

    answer = robot_listen(prompt, pepper).lower()

    if "interrupt" in answer:
        raise KeyboardInterrupt("User interrupted the interaction.")

    if any(w in answer for w in ["no", "not now", "later", "stop", "cancel"]):
        robot_say("No problem. I will keep looking around in case someone else needs help.", pepper)
        return False

    return True

def ask_save_new_user_after_interaction(name: str, interest: str = "Unknown", pepper=None) -> None:
    """
    Ask a new user at the end whether SocioBot may remember them.
    This is more natural than asking before the conversation starts.
    """
    if not has_pending_new_user_encoding():
        return

    if not name or name in ("Guest", "unidentified"):
        robot_say(
            "I did not catch your name clearly, so I will not save your face data this time.",
            pepper
        )
        discard_pending_new_user_encoding()
        return

    prompt = (
        f"Before you go, should I remember you as {name} "
        "so I can recognize you next time?"
    )

    for attempt in range(2):
        answer = robot_listen(prompt, pepper)

        if "interrupt" in answer.lower():
            discard_pending_new_user_encoding()
            raise KeyboardInterrupt("User interrupted during save consent.")

        if _is_no_response(answer):
            robot_say(
                "No problem. I will not save your face data.",
                pepper
            )
            discard_pending_new_user_encoding()
            return

        if _is_yes_response(answer):
            saved = save_pending_new_user(name, interest)
            if saved:
                robot_say(
                    f"Done. I will remember you for next time, {name}.",
                    pepper
                )
            else:
                robot_say(
                    "I could not save the profile this time.",
                    pepper
                )
            return

        prompt = "I just need a clear yes or no. Should I remember you for next time?"

    robot_say(
        "No problem. I will not save your face data.",
        pepper
    )
    discard_pending_new_user_encoding()


class SocioBotApp:
    """Coordinates the full SocioBot interaction as an object."""

    def __init__(self, pepper=None, sim_manager=None):
        self.pepper = pepper
        self.sim_manager = sim_manager
        self.robot = RobotInteractionLayer(pepper)
        self.robot.sim_manager = sim_manager
        self.face_recognizer = FaceRecognizer(pepper)
        self.recommender = EventRecommendationNetwork()

    def present_recommendations(self, ranking: list, inferred: dict) -> str:
        return present_recommendations(ranking, inferred, self.pepper)

    def farewell_sequence(self, name: str, top_event: str) -> None:
        farewell_sequence(name, top_event, self.pepper)

    def introduce(self, status: str, name: str) -> None:
        introduce_sociobot(status, name, self.pepper)

    def ask_recommendation_consent(self, status: str, name: str) -> bool:
        return ask_recommendation_consent(status, name, self.pepper)

    def ask_save_new_user_after_interaction(
        self,
        name: str,
        interest: str = "Unknown",
    ) -> None:
        ask_save_new_user_after_interaction(name, interest, self.pepper)

    def run_interaction(self) -> bool:
        """Execute one full user interaction from recognition to farewell."""

        # Face recognition
        print("\n[STEP 1] Running face recognition...")
        status, name = self.face_recognizer.run()

        print(f"\n Recognition result: status={status!r}, name={name!r}")

        if status == "manual_stop":
            self.robot.say("Okay, I will stop now.")
            return False

        if status == "unknown":
            self.robot.say(
                "I could not complete the recognition this time. "
                "I will keep looking for someone who needs event ideas."
            )
            return True

        # Build initial evidence from face recognition
        print("\n[STEP 2] Building initial evidence...")
        evidence: dict[str, str] = {}

        if status == "known":
            evidence["RecognizedUser"] = "Known"
            history = get_preference_history(name)
            evidence["UserPreferenceHistory"] = history
            print(f"  - Known user. History preference: {history}")
            self.introduce(status, name)
        elif status == "new":
            evidence["RecognizedUser"]        = "Unknown"
            evidence["UserPreferenceHistory"] = "Unknown"
            print("  - New user. No preference history available.")

        if not self.ask_recommendation_consent(status, name):
            if status == "new":
                self.ask_save_new_user_after_interaction(name, "Unknown")
            return True

        time.sleep(0.5)

        # Dialog — collect preferences
        print("\n[STEP 3] Running preference dialog...")
        dm = DialogManager(pepper=self.pepper)
        preference_evidence = dm.run()
        evidence.update(preference_evidence)

        print("\n  - Final evidence dict:")
        for node, value in evidence.items():
            print(f"     {node:<26} = {value}")

        # Bayesian Network inference
        print("\n[STEP 4] Running Bayesian inference...")
        self.robot.think(duration=2.5)

        try:
            ranking, inferred = self.recommender.recommend(evidence)
            print("  Inference complete.")
        except Exception as e:
            print(f"  Inference error: {e}")
            self.robot.say(
                "I'm sorry, something went wrong with my reasoning engine. "
                "Please try again!"
            )
            return True

        # Present recommendations
        print("\n[STEP 5] Presenting recommendations...")
        top_event = self.present_recommendations(ranking, inferred)
        top_interest = max(
            inferred["InterestType"], key=inferred["InterestType"].get
        )

        # Save user interest for personalisation
        if status == "known" and name not in ("unidentified", "Guest"):
            save_user_preference(name, top_interest)
            print(f" Saved interest '{top_interest}' for user '{name}'.")

        elif status == "new":
            self.ask_save_new_user_after_interaction(name, top_interest)

        # Farewell
        print("\n[STEP 6] Farewell sequence...")
        self.farewell_sequence(name, top_event)
        print("\n Interaction complete. Resetting...\n")

        return True

    def shutdown(self) -> None:
        print("\n[SHUTDOWN] Cleaning up...")
        if self.sim_manager is not None:
            try:
                manager, client_id = self.sim_manager
                manager.stopSimulation(client_id)
                print("[SHUTDOWN] qiBullet stopped.")
            except Exception as e:
                print(f"[WARN] Simulation stop error: {e}")
        print("[SHUTDOWN] SocioBot offline. Goodbye!")


#  MAIN INTERACTION LOOP
def run_interaction(pepper=None) -> bool:
    return SocioBotApp(pepper=pepper).run_interaction()


#  ENTRY POINT
def parse_args():
    parser = argparse.ArgumentParser(description="SocioBot — Social Event Recommendation Robot")
    parser.add_argument(
        "--no-robot", action="store_true",
        help="Run without qiBullet simulation (text-only mode)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║              SocioBot                    ║")
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

    app = SocioBotApp(pepper=pepper, sim_manager=sim_manager)

    # Main loop — keep running until keyboard interrupt
    try:
        while True:
            keep_running = app.run_interaction()
            if not keep_running:
                break

    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user.")

    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
