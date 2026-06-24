from __future__ import annotations

"""
conversation_module.py
-----------------------
Guided conversation layer for the Preference-Based Social Event Recommendation Robot.

Sits between face_recognition_module and the Bayesian network.
After recognition, Pepper asks the user a short sequence of questions,
maps the answers to BN evidence nodes, runs inference, and announces
the top recommended events.

Usage:
    from conversation_module import run_conversation

    # After face recognition:
    status, name = get_user_status(pepper)
    run_conversation(status, name, pepper)
"""

import time
from event_recommendation_bn import recommend_events, STATES

# ── Question definitions ──────────────────────────────────────────────────────
# Each entry maps a BN node to a spoken question + answer choices.
# 'key'      : BN evidence node name (must match STATES keys)
# 'question' : what Pepper says aloud
# 'choices'  : dict mapping spoken/typed keyword -> BN state label
# 'tip'      : short hint shown in the terminal listing valid options

QUESTIONS = [
    {
        "key": "Mood",
        "question": "How are you feeling right now? Relaxed, excited, or a bit stressed?",
        "choices": {
            "relaxed": "Relaxed",
            "calm":    "Relaxed",
            "chill":   "Relaxed",
            "excited": "Excited",
            "happy":   "Excited",
            "energetic": "Excited",
            "stressed": "Stressed",
            "tired":   "Stressed",
            "anxious": "Stressed",
        },
        "tip": "relaxed / excited / stressed",
    },
    {
        "key": "EnvironmentPreference",
        "question": "Would you prefer an indoor or outdoor event? Or no preference?",
        "choices": {
            "indoor":       "Indoor",
            "inside":       "Indoor",
            "outdoor":      "Outdoor",
            "outside":      "Outdoor",
            "no preference": "NoPreference",
            "no":           "NoPreference",
            "either":       "NoPreference",
            "both":         "NoPreference",
        },
        "tip": "indoor / outdoor / no preference",
    },
    {
        "key": "GroupSize",
        "question": (
            "Are you coming alone, with one friend, a small group, or a large group?"
        ),
        "choices": {
            "alone":        "Alone",
            "solo":         "Alone",
            "just me":      "Alone",
            "pair":         "Pair",
            "two":          "Pair",
            "friend":       "Pair",
            "small":        "SmallGroup",
            "small group":  "SmallGroup",
            "few":          "SmallGroup",
            "large":        "LargeGroup",
            "large group":  "LargeGroup",
            "many":         "LargeGroup",
            "everyone":     "LargeGroup",
        },
        "tip": "alone / pair / small group / large group",
    },
    {
        "key": "BudgetPreference",
        "question": "What is your budget for tonight? Free, low, medium, or high?",
        "choices": {
            "free":    "Free",
            "nothing": "Free",
            "zero":    "Free",
            "low":     "Low",
            "cheap":   "Low",
            "little":  "Low",
            "medium":  "Medium",
            "moderate": "Medium",
            "some":    "Medium",
            "high":    "High",
            "plenty":  "High",
            "a lot":   "High",
        },
        "tip": "free / low / medium / high",
    },
    {
        "key": "UserPreferenceHistory",
        "question": (
            "What kind of events do you enjoy most? "
            "Music, food, sports, culture, networking, or games?"
        ),
        "choices": {
            "music":       "Music",
            "concert":     "Music",
            "songs":       "Music",
            "food":        "Food",
            "eating":      "Food",
            "restaurant":  "Food",
            "sports":      "Sports",
            "sport":       "Sports",
            "athletic":    "Sports",
            "culture":     "Culture",
            "art":         "Culture",
            "museum":      "Culture",
            "networking":  "Networking",
            "meeting":     "Networking",
            "professional": "Networking",
            "games":       "Games",
            "gaming":      "Games",
            "board games": "Games",
        },
        "tip": "music / food / sports / culture / networking / games",
    },
]

# For known (returning) users we can skip the interest question and
# use their stored history. This flag controls that behaviour.
SKIP_INTEREST_FOR_KNOWN_USER = True


# ── Answer parsing ────────────────────────────────────────────────────────────

def _parse_answer(raw: str, choices: dict) -> str | None:
    """
    Match the user's free-text answer against the choice keywords.
    Returns the BN state label or None if no match is found.
    """
    lowered = raw.lower().strip()
    # Exact substring match first (longest key wins to avoid "no" matching "no preference")
    matched = None
    matched_len = 0
    for keyword, label in choices.items():
        if keyword in lowered and len(keyword) > matched_len:
            matched = label
            matched_len = len(keyword)
    return matched


# ── Single question loop ──────────────────────────────────────────────────────

def _ask_question(q: dict, pepper=None, retries: int = 2) -> str:
    """
    Ask one BN question, retry on unrecognised answers.
    Returns the matched BN state label, or falls back to the first state
    in the BN node's state list (a safe default).
    """
    from face_recognition_module_with_Robot import robot_say, robot_listen  # local import to avoid circular deps

    for attempt in range(retries + 1):
        raw = robot_listen(q["question"], pepper)

        label = _parse_answer(raw, q["choices"])
        if label:
            robot_say("Got it, {}.".format(label), pepper)
            return label

        if attempt < retries:
            robot_say(
                "Sorry, I did not catch that. Please say one of: {}.".format(q["tip"]),
                pepper,
            )

    # Fallback: pick the middle state so inference is not dominated by the edge
    fallback = STATES[q["key"]][len(STATES[q["key"]]) // 2]
    robot_say(
        "I will assume {} for now.".format(fallback), pepper
    )
    return fallback


# ── Survey orchestration ──────────────────────────────────────────────────────

def collect_preferences(
    status: str,
    name: str,
    known_history: str | None = None,
    pepper=None,
) -> dict:
    """
    Run the spoken survey and return a complete evidence dictionary
    ready to pass to recommend_events().

    Parameters
    ----------
    status        : "known" | "new" | "unknown"
    name          : user's name (for personalised phrasing)
    known_history : stored UserPreferenceHistory label for returning users
                    (load from your user DB; pass None to ask anyway)
    pepper        : PepperVirtual instance or None

    Returns
    -------
    evidence dict e.g. {
        "RecognizedUser": "Known",
        "UserPreferenceHistory": "Games",
        "Mood": "Relaxed",
        "EnvironmentPreference": "Indoor",
        "GroupSize": "SmallGroup",
        "BudgetPreference": "Low",
    }
    """
    from face_recognition_module_with_Robot import robot_say  # local import

    evidence: dict = {}

    # ── RecognizedUser evidence ───────────────────────────────────────────────
    evidence["RecognizedUser"] = "Known" if status == "known" else "Unknown"

    # ── Opening line ──────────────────────────────────────────────────────────
    if status == "known":
        robot_say(
            "Great to see you again, {}! "
            "Let me ask a few quick questions so I can find the perfect event for you tonight.".format(name),
            pepper,
        )
    else:
        robot_say(
            "Nice to meet you, {}! "
            "I will ask you a few questions to recommend the best event for you.".format(name),
            pepper,
        )

    time.sleep(0.4)

    # ── Ask each question in order ────────────────────────────────────────────
    for q in QUESTIONS:
        node = q["key"]

        # Skip the interest question for known users if we have stored history
        if (
            node == "UserPreferenceHistory"
            and status == "known"
            and SKIP_INTEREST_FOR_KNOWN_USER
            and known_history is not None
        ):
            evidence[node] = known_history
            robot_say(
                "Based on your history I know you enjoy {}. I will use that.".format(known_history),
                pepper,
            )
            continue

        evidence[node] = _ask_question(q, pepper)
        time.sleep(0.2)

    return evidence


# ── Announce results ──────────────────────────────────────────────────────────

def announce_recommendations(
    ranking: list[tuple[str, float]],
    inferred: dict,
    name: str,
    pepper=None,
    top_n: int = 3,
) -> None:
    """
    Have Pepper announce the top-N recommended events with short explanations.
    """
    from face_recognition_module_with_Robot import robot_say, robot_wave  # local import

    # Derive dominant inferred states for a natural explanation
    social_energy = max(inferred["SocialEnergy"], key=inferred["SocialEnergy"].get)
    interest_type = max(inferred["InterestType"], key=inferred["InterestType"].get)

    robot_say(
        "Thanks, {}! Based on your answers, it looks like you have {} social energy "
        "and a leaning toward {} events.".format(name, social_energy.lower(), interest_type.lower()),
        pepper,
    )
    time.sleep(0.5)

    robot_say("Here are my top recommendations for tonight:", pepper)
    time.sleep(0.3)

    # ── Friendly descriptions for each event ─────────────────────────────────
    descriptions = {
        "Concert":        "a live music concert",
        "FoodFestival":   "a food festival",
        "SportsEvent":    "a sports event",
        "MuseumVisit":    "a museum visit",
        "StudentMeetup":  "a student meetup",
        "BoardGameNight": "a board game night",
        "Picnic":         "an outdoor picnic",
    }

    for rank, (event, prob) in enumerate(ranking[:top_n], start=1):
        label = descriptions.get(event, event)
        pct   = int(prob * 100)
        robot_say(
            "Number {}. {}. Confidence: {} percent.".format(rank, label.capitalize(), pct),
            pepper,
        )
        time.sleep(0.4)

    robot_say("I hope you have a wonderful time tonight, {}!".format(name), pepper)
    robot_wave(pepper)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_conversation(
    status: str,
    name: str,
    pepper=None,
    known_history: str | None = None,
) -> dict:
    """
    Full pipeline: survey → BN inference → announcement.

    Parameters
    ----------
    status        : "known" | "new" | "unknown"  (from face_recognition_module)
    name          : user's name
    pepper        : PepperVirtual instance or None
    known_history : pre-stored UserPreferenceHistory for returning users

    Returns
    -------
    dict with keys 'evidence', 'ranking', 'inferred_nodes'
    """
    from face_recognition_module_with_Robot import robot_say  # local import

    if status == "unknown":
        robot_say("I could not identify you. Have a great day!", pepper)
        return {}

    # Step 1: collect survey answers → evidence
    evidence = collect_preferences(status, name, known_history, pepper)

    # Step 2: run Bayesian inference
    print("\n[BN] Running inference with evidence:")
    for k, v in evidence.items():
        print("     {} = {}".format(k, v))

    ranking, inferred_nodes = recommend_events(evidence)

    print("\n[BN] Top recommendations:")
    for event, prob in ranking:
        print("     {}: {:.3f}".format(event, prob))

    # Step 3: Pepper announces results
    announce_recommendations(ranking, inferred_nodes, name, pepper)

    return {
        "evidence":       evidence,
        "ranking":        ranking,
        "inferred_nodes": inferred_nodes,
    }


# ── Standalone test (no robot needed) ────────────────────────────────────────
if __name__ == "__main__":
    # Simulate a known returning user without launching qiBullet
    print("=== Conversation Module — terminal test ===\n")

    result = run_conversation(
        status="known",
        name="Alice",
        pepper=None,               # no robot — keyboard fallback
        known_history="Games",     # Alice's stored preference
    )

    print("\n=== Final result ===")
    for event, prob in result.get("ranking", []):
        print("  {}: {:.3f}".format(event, prob))