"""
Conversational preference elicitation for EventMate.

Asks the user 4 questions via robot_say / robot_listen (gestures.py),
maps natural-language answers to BN state names, and returns the
evidence dict ready to pass to recommend_events() in
event_recommendation_bn.py.

BN nodes collected here:
  Mood                  : Relaxed | Excited | Stressed
  EnvironmentPreference : Indoor | Outdoor | NoPreference
  GroupSize             : Alone | Pair | SmallGroup | LargeGroup
  BudgetPreference      : Free | Low | Medium | High

RecognizedUser and UserPreferenceHistory are passed in from
face_recognition.py — they are not asked during dialog.

Run standalone to test without robot:
  python modules/dialog_manager.py
"""

from __future__ import annotations
import re
from typing import Optional

try:
    from modules.gestures import robot_act_and_say, robot_say, robot_listen, robot_nod
except ModuleNotFoundError:
    from gestures import robot_act_and_say, robot_say, robot_listen, robot_nod

#  PROFANITY FILTER
_BANNED = {
    "fuck", "shit", "ass", "bitch", "bastard", "crap",
    "damn", "piss", "cock", "dick", "pussy", "cunt",
    "idiot", "stupid", "moron", "retard", "hate"
}

def is_abusive(text: str) -> bool:
    """Return True if text contains any banned word."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    return bool(words & _BANNED)

#  ANSWER → BN STATE MAPPERS
def _map_mood(answer: str) -> Optional[str]:
    """Map free-text mood answer to BN state."""
    a = answer.lower()
    if any(w in a for w in ["relax", "calm", "chill", "tired", "quiet", "peaceful", "easy"]):
        return "Relaxed"
    if any(w in a for w in ["excit", "happy", "energet", "fun", "great", "amaz", "hype", "pumped"]):
        return "Excited"
    if any(w in a for w in ["stress", "anxious", "nervous", "busy", "overwhelm", "worried", "bad"]):
        return "Stressed"
    return None

def _map_environment(answer: str) -> Optional[str]:
    """Map free-text environment answer to BN state."""
    a = answer.lower()
    if any(w in a for w in ["outdoor", "outside", "open air", "nature", "park", "fresh"]):
        return "Outdoor"
    if any(w in a for w in ["indoor", "inside", "cozy", "cosy", "warm", "shelter"]):
        return "Indoor"
    if any(w in a for w in ["no preference", "either", "any", "don't mind", "dont mind", "both"]):
        return "NoPreference"
    return None

def _map_group_size(answer: str) -> Optional[str]:
    """Map free-text group size answer to BN state."""
    a = answer.lower()
    if any(w in a for w in ["alone", "solo", "myself", "just me", "by myself", "1"]):
        return "Alone"
    if any(w in a for w in ["pair", "partner", "friend", "date", "two", "2"]):
        return "Pair"
    if any(w in a for w in ["small group", "small", "few", "3", "4", "handful"]):
        return "SmallGroup"
    if any(w in a for w in ["large group", "large", "big", "many", "lots", "crowd", "5", "6", "7", "8"]):
        return "LargeGroup"
    
    # Fallback: count numeric mentions
    nums = re.findall(r"\d+", answer)
    if nums:
        n = int(nums[0])
        if n == 1:
            return "Alone"
        elif n == 2:
            return "Pair"
        elif n <= 4:
            return "SmallGroup"
        else:
            return "LargeGroup"
    return None

def _map_budget(answer: str) -> Optional[str]:
    """Map free-text budget answer to BN state."""
    a = answer.lower()
    if any(w in a for w in ["free", "no cost", "nothing", "zero", "gratis", "no money"]):
        return "Free"
    if any(w in a for w in ["low", "cheap", "budget", "little", "minimal", "affordable", "10", "15", "20"]):
        return "Low"
    if any(w in a for w in ["medium", "moderate", "average", "mid", "reasonable", "30", "40", "50"]):
        return "Medium"
    if any(w in a for w in ["high", "lot", "expensive", "generous", "plenty", "rich", "100", "unlimited"]):
        return "High"
    return None


#  DIALOG QUESTIONS
# Each entry: (question_text, mapper_fn, BN_node_name, valid_states, hint)
QUESTIONS = [
    (
        "How are you feeling right now? For example: relaxed, excited, or a bit stressed?",
        _map_mood,
        "Mood",
        ["Relaxed", "Excited", "Stressed"],
        "You can say things like calm, happy, or tired."
    ),
    (
        "Do you prefer indoor or outdoor activities? Or do you have no preference?",
        _map_environment,
        "EnvironmentPreference",
        ["Indoor", "Outdoor", "NoPreference"],
        "Try saying indoor, outdoor, or either."
    ),
    (
        "Are you going alone, with one friend, a small group, or a large group?",
        _map_group_size,
        "GroupSize",
        ["Alone", "Pair", "SmallGroup", "LargeGroup"],
        "You can say alone, pair, small group, or large group."
    ),
    (
        "What is your budget for today? Free, low, medium, or high?",
        _map_budget,
        "BudgetPreference",
        ["Free", "Low", "Medium", "High"],
        "For example: free, low budget, medium, or I can spend a lot."
    ),
]


#  DIALOG MANAGER CLASS
class DialogManager:
    """Manages the preference elicitation conversation"""

    MAX_RETRIES = 2   # re-ask a misunderstood question

    def __init__(self, pepper=None):
        self.pepper = pepper
        self.evidence: dict[str, str] = {}

    def run(self) -> dict[str, str]:
        """
        Run the full preference survey and return collected evidence.
        """
        robot_say(
            "Great! I have a few quick questions to find the best event for you.",
            self.pepper
        )

        for question_text, mapper, node_name, valid_states, hint in QUESTIONS:
            value = self._ask_question(question_text, mapper, node_name, hint)
            self.evidence[node_name] = value

        robot_say("Perfect! I have everything I need. Let me think...", self.pepper)
        return self.evidence

    def _ask_question(
        self,
        question: str,
        mapper,
        node_name: str,
        hint: str
    ) -> str:
        """
        Ask a single question, retrying up to MAX_RETRIES times on parse failure.
        Falls back to the first valid state if still unresolved.
        """
        valid_states = {k for _, _, n, v, _ in QUESTIONS if n == node_name for k in v}
        # Build valid_states from the QUESTIONS list entry
        entry = next((e for e in QUESTIONS if e[2] == node_name), None)
        fallback = entry[3][0] if entry else "Unknown"

        for attempt in range(self.MAX_RETRIES + 1):
            answer = robot_listen(question, self.pepper)

            # Profanity check
            if is_abusive(answer):
                robot_say(
                    "I am sorry, but I cannot process that kind of language. "
                    "Please keep the conversation friendly!",
                    self.pepper
                )
                continue  # re-ask same question, don't count as a retry

            # Map answer to BN state
            mapped = mapper(answer)

            if mapped is not None:
                robot_act_and_say(
                    robot_nod,
                    f"Got it — {mapped}.",
                    self.pepper,
                    speech_delay=0.1,
                )
                return mapped

            # Could not parse
            if attempt < self.MAX_RETRIES:
                robot_say(
                    f"I did not quite catch that. {hint}",
                    self.pepper
                )
            else:
                robot_say(
                    f"No problem, I will assume {fallback} for now.",
                    self.pepper
                )
                return fallback

        return fallback  # safety return (should not reach here)

    def reset(self) -> None:
        """Clear collected evidence for a new interaction."""
        self.evidence = {}


#  STANDALONE TEST (no robot, keyboard only)
if __name__ == "__main__":
    print("=== Dialog Manager Standalone Test ===\n")
    dm = DialogManager(pepper=None)
    evidence = dm.run()
    print("\nCollected evidence:")
    for node, value in evidence.items():
        print(f"  {node}: {value}")
