"""
Conversational preference elicitation for EventMate
"""
from __future__ import annotations

import re
import random
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


class ProfanityFilter:
    """Checks dialog input against the existing banned-word list."""

    def __init__(self, banned_words=None):
        self.banned_words = set(banned_words or _BANNED)

    def is_abusive(self, text: str) -> bool:
        words = set(re.findall(r"[a-z]+", text.lower()))
        return bool(words & self.banned_words)


def is_abusive(text: str) -> bool:
    return ProfanityFilter().is_abusive(text)


class AnswerMapper:
    """Maps free-text answers into Bayesian Network state names."""

    @staticmethod
    def map_mood(answer: str) -> Optional[str]:
        return _map_mood(answer)

    @staticmethod
    def map_environment(answer: str) -> Optional[str]:
        return _map_environment(answer)

    @staticmethod
    def map_group_size(answer: str) -> Optional[str]:
        return _map_group_size(answer)

    @staticmethod
    def map_budget(answer: str) -> Optional[str]:
        return _map_budget(answer)

#  ANSWER --> BN STATE MAPPERS
def _map_mood(answer: str) -> Optional[str]:
    """Map free-text mood answer to BN state, including vague/indirect phrasing."""

    a = answer.lower()
    if any(w in a for w in ["relax", "calm", "chill", "tired", "quiet", "peaceful", "easy"]):
        return "Relaxed"
    if any(w in a for w in ["excit", "happy", "energet", "fun", "great", "amaz", "hype", "pumped", "thrilled"]):
        return "Excited"
    if any(w in a for w in ["stress", "anxious", "nervous", "overwhelm", "worried",
                            "not great", "not good", "not well", "bad", "terrible", "awful"]):
        return "Stressed"
    
    # Mildly-positive / neutral phrasing ("I'm alright", "feeling well", "doing fine")
    # implies a generally good but non-hyped mood -> closer to Relaxed than Excited.
    if any(w in a for w in ["alright", "all right", "fine", "okay", "ok", "good", "well",
                             "not bad", "pretty good", "decent", "content", "fair"]):
        # But upgrade to Excited if paired with an enthusiasm cue.
        EXCITED_BOOSTERS = {"really excited", "very excited", "super pumped", "hyped", "fantastic", "awesome", "amazing"}
        if any(phrase in a for phrase in EXCITED_BOOSTERS):
            return "Excited"
        return "Relaxed"
    return None

def _map_environment(answer: str) -> Optional[str]:
    """Map free-text environment answer to BN state, including vague phrasing."""
    a = answer.lower()
    if any(w in a for w in ["outdoor", "outside", "open air", "nature", "park", "fresh"]):
        return "Outdoor"
    if any(w in a for w in ["indoor", "inside", "cozy", "cosy", "warm", "shelter"]):
        return "Indoor"
    if any(w in a for w in ["no preference", "either", "any", "don't mind", "dont mind",
                            "both", "doesn't matter", "doesnt matter", "whatever", "not sure"]):
        return "NoPreference"
    return None

def _map_group_size(answer: str) -> Optional[str]:
    """Map free-text group size answer to BN state, including vague phrasing."""
    a = answer.lower()
    if any(w in a for w in ["alone", "solo", "myself", "just me", "by myself", "on my own", "1"]):
        return "Alone"
    if any(w in a for w in ["small group", "small", "few", "3", "4", "handful", "couple of friends"]):
            return "SmallGroup"
    if any(w in a for w in ["pair", "partner", "friend", "date", "two", "couple", "girlfriend", "boyfriend", "2"]):
        return "Pair"
    if any(w in a for w in ["large group", "large", "big", "many", "lots", "crowd", "everyone", "whole gang", "5", "6", "7", "8"]):
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
    """Map free-text budget answer to BN state, including vague phrasing."""
    a = answer.lower()
    if any(w in a for w in ["free", "no cost", "nothing", "zero", "gratis", "no money", "can't spend", "cant spend"]):
        return "Free"
    if any(w in a for w in ["low", "cheap", "budget", "little", "minimal", "affordable",
                            "tight", "not much", "small budget", "10", "15", "20"]):
        return "Low"
    if any(w in a for w in ["medium", "moderate", "average", "mid", "reasonable",
                            "some money", "okay budget", "30", "40", "50"]):
        return "Medium"
    if any(w in a for w in ["high", "lot", "expensive", "generous", "plenty", "rich",
                            "no limit", "money is not an issue", "spend a lot", "100", "unlimited"]):
        return "High"
    return None


#  DIALOG QUESTIONS
# Each entry: (question_text, mapper_fn, BN_node_name, valid_states, hint)
QUESTIONS = [
    (
        [
            "How are you doing today?",
            "How has your day been so far?",
            "What kind of mood are you in right now?",
            "Before I suggest anything, how are you feeling today?"
        ],
        AnswerMapper.map_mood,
        "Mood",
        ["Relaxed", "Excited", "Stressed"],
        "You can answer naturally, like calm, excited, tired, stressed, or pretty good."
    ),
    (
        [
            "Do you feel like being indoors, outside, or are you open to either?",
            "Would you rather do something indoors or outdoors today?",
            "Are you in the mood for a cozy indoor plan, something outside, or either one?",
            "What kind of setting sounds better today: inside, outside, or no strong preference?"
        ],
        AnswerMapper.map_environment,
        "EnvironmentPreference",
        ["Indoor", "Outdoor", "NoPreference"],
        "You can say indoor, outdoor, either, both, or I do not mind."
    ),
    (
        [
            "Who are you planning to go with?",
            "Is this just for you, or are other people joining?",
            "Are you going alone, with one person, or with a group?",
            "How many people should I keep in mind for the recommendation?"
        ],
        AnswerMapper.map_group_size,
        "GroupSize",
        ["Alone", "Pair", "SmallGroup", "LargeGroup"],
        "You can say alone, with a friend, two people, a few friends, or a big group."
    ),
    (
        [
            "How much would you like to spend?",
            "Should I keep it free or low-cost, or is a paid event okay?",
            "What kind of budget should I consider?",
            "Do you want something cheap today, or are you okay spending a bit more?"
        ],
        AnswerMapper.map_budget,
        "BudgetPreference",
        ["Free", "Low", "Medium", "High"],
        "You can say free, cheap, low budget, moderate, or I can spend more."
    ),
]

RETRY_QUESTIONS = {
    "Mood": [
        "Let me put it another way. Are you feeling more relaxed, energetic, or stressed?",
        "Would you say your mood is calm, excited, or a bit overwhelmed?"
    ],
    "EnvironmentPreference": [
        "Would you prefer something inside, something outside, or either one?",
        "Should I look for indoor plans, outdoor plans, or keep both open?"
    ],
    "GroupSize": [
        "How many people should I plan for: just you, two people, a small group, or a larger group?",
        "Is this a solo plan, a date, a few friends, or a bigger group?"
    ],
    "BudgetPreference": [
        "Should I keep it free, low-cost, moderate, or is spending more okay?",
        "What budget range should I use: free, cheap, moderate, or high?"
    ],
}

def _natural_ack(node_name: str, mapped: str) -> str:
    """Return a natural acknowledgement instead of repeating BN state names."""
    responses = {
        "Mood": {
            "Relaxed": [
                "Okay, sounds like you want something easygoing.",
                "Got it. I will keep the vibe more relaxed."
            ],
            "Excited": [
                "Nice, then I can look for something with more energy.",
                "Great, I will keep livelier options in mind."
            ],
            "Stressed": [
                "Okay, then I will avoid anything too intense.",
                "Understood. I will look for something that feels lighter."
            ],
        },
        "EnvironmentPreference": {
            "Indoor": [
                "Indoor sounds good.",
                "Alright, I will focus more on indoor options."
            ],
            "Outdoor": [
                "Outdoor it is.",
                "Nice, I will keep outdoor plans in mind."
            ],
            "NoPreference": [
                "Okay, I will keep both indoor and outdoor options open.",
                "That gives us some flexibility."
            ],
        },
        "GroupSize": {
            "Alone": [
                "Got it, I will look for something that also works solo.",
                "Okay, I will keep it comfortable for one person."
            ],
            "Pair": [
                "Nice, I will think of options that work well for two.",
                "Okay, something suitable for you and one other person."
            ],
            "SmallGroup": [
                "Great, I will look for something that works for a small group.",
                "A few people, got it."
            ],
            "LargeGroup": [
                "Sounds like a bigger group. I will keep that in mind.",
                "Alright, I will focus on options that can handle more people."
            ],
        },
        "BudgetPreference": {
            "Free": [
                "Okay, I will prioritize free options.",
                "Got it, no-cost options first."
            ],
            "Low": [
                "I will keep it budget-friendly.",
                "Alright, I will avoid expensive options."
            ],
            "Medium": [
                "Okay, a reasonable budget gives us a few good choices.",
                "Got it, I can include moderately priced events."
            ],
            "High": [
                "Great, then I do not need to restrict the options too much.",
                "Okay, I can consider more premium events too."
            ],
        },
    }

    return random.choice(
        responses.get(node_name, {}).get(mapped, [f"Got it, {mapped}."])
    )


#  DIALOG MANAGER CLASS
class DialogManager:
    """Manages the preference elicitation conversation"""

    MAX_RETRIES = 2   # re-ask a misunderstood question

    def __init__(self, pepper=None, profanity_filter: Optional[ProfanityFilter] = None):
        self.pepper = pepper
        self.evidence: dict[str, str] = {}
        self.profanity_filter = profanity_filter or ProfanityFilter()

    def run(self) -> dict[str, str]:
        """
        Run the full preference survey and return collected evidence.
        """
        robot_say(
            random.choice([
                "Great. I will ask a few quick things, then I can suggest something that actually fits you.",
                "Perfect. Let me get a quick sense of your mood and plans first.",
                "Nice. I just need a little context before I recommend something."
            ]),
            self.pepper
        )

        for question_text, mapper, node_name, valid_states, hint in QUESTIONS:
            question = random.choice(question_text) if isinstance(question_text, list) else question_text
            value = self._ask_question(question, mapper, node_name, hint)
            self.evidence[node_name] = value

        robot_say(
            random.choice([
                "That helps. Give me a moment to think through the options.",
                "Okay, I have enough to work with. Let me check what fits best.",
                "Great, I can make a recommendation from that."
            ]),
            self.pepper
        )
        return self.evidence

    def _ask_question(
        self,
        question: str,
        mapper,
        node_name: str,
        hint: str
    ) -> str:
        """
        Ask a single question, retrying with rephrased clarification prompts
        """
        entry = next((e for e in QUESTIONS if e[2] == node_name), None)
        fallback = entry[3][0] if entry else "Unknown"

        current_prompt = question

        for attempt in range(self.MAX_RETRIES + 1):
            answer = robot_listen(current_prompt, self.pepper)

            if "interrupt" in answer.lower():
                robot_say("Okay, I will stop here.", self.pepper)
                raise KeyboardInterrupt("User interrupted the dialog.")

            if self.profanity_filter.is_abusive(answer):
                robot_say(
                    "I am sorry, but I cannot process that kind of language. "
                    "Please keep the conversation friendly.",
                    self.pepper
                )
                current_prompt = random.choice(RETRY_QUESTIONS.get(node_name, [hint]))
                continue

            mapped = mapper(answer)

            if mapped is not None:
                robot_act_and_say(
                    robot_nod,
                    _natural_ack(node_name, mapped),
                    self.pepper,
                    speech_delay=0.1,
                )
                return mapped

            if attempt < self.MAX_RETRIES:
                current_prompt = random.choice(RETRY_QUESTIONS.get(node_name, [hint]))
            else:
                robot_say(
                    f"No problem, I will assume {fallback} for now.",
                    self.pepper
                )
                return fallback

        return fallback

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
