from __future__ import annotations

import os
import subprocess
import itertools
import pyAgrum as gum


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEMO_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "demo_output")


STATES = {
    "RecognizedUser": ["Known", "Unknown"],

    "UserPreferenceHistory": [
        "Music", "Food", "Sports", "Culture", "Networking", "Games", "Unknown"
    ],

    "Mood": ["Relaxed", "Excited", "Stressed"],

    "EnvironmentPreference": ["Indoor", "Outdoor", "NoPreference"],

    "GroupSize": ["Alone", "Pair", "SmallGroup", "LargeGroup"],

    "BudgetPreference": ["Free", "Low", "Medium", "High"],

    # Inference nodes
    "SocialEnergy":    ["Low", "Medium", "High"],
    "TimeRestrictions": ["Short", "Medium", "Long"],

    "InterestType": [
        "Music", "Food", "Sports", "Culture", "Networking", "Games"
    ],

    "EventScoreMatching": ["Low", "Medium", "High"],

    # Output node
    "RecommendedEvent": [
        "Concert", "FoodFestival", "SportsEvent",
        "MuseumVisit", "StudentMeetup", "BoardGameNight", "Picnic"
    ]
}


BN_ARCS = [
    ("RecognizedUser",       "UserPreferenceHistory"),

    ("UserPreferenceHistory", "Mood"),
    ("UserPreferenceHistory", "EventScoreMatching"),

    ("Mood",                 "SocialEnergy"),
    ("EnvironmentPreference","SocialEnergy"),
    ("GroupSize",            "SocialEnergy"),

    ("EnvironmentPreference","TimeRestrictions"),
    ("GroupSize",            "TimeRestrictions"),

    ("SocialEnergy",         "InterestType"),
    ("TimeRestrictions",     "InterestType"),
    ("GroupSize",            "InterestType"),
    ("BudgetPreference",     "InterestType"),

    ("InterestType",         "EventScoreMatching"),

    ("InterestType",         "RecommendedEvent"),   # <-- NEW arc
    ("EventScoreMatching",   "RecommendedEvent"),
]


#  UTILITY FUNCTIONS
def add_variable(bn, name, labels):
    var = gum.LabelizedVariable(name, name, 0)
    for label in labels:
        var.addLabel(label)
    return bn.add(var)


def normalize(values):
    total = sum(values)
    if total == 0:
        return [1.0 / len(values)] * len(values)
    return [v / total for v in values]


#  CPT FUNCTIONS 
def social_energy_probs(mood, environment, group_size):
    score = 0
    if mood == "Excited":    score += 2
    elif mood == "Stressed": score -= 2

    if environment == "Outdoor": score += 1

    if group_size == "Alone":      score -= 1
    elif group_size == "SmallGroup": score += 1
    elif group_size == "LargeGroup": score += 2

    if score >= 3:   return [0.05, 0.25, 0.70]
    elif score == 2: return [0.10, 0.40, 0.50]
    elif score == 1: return [0.20, 0.55, 0.25]
    elif score == 0: return [0.30, 0.50, 0.20]
    elif score == -1:return [0.50, 0.40, 0.10]
    else:            return [0.75, 0.20, 0.05]


def time_restriction_probs(environment, group_size):
    score = 0
    if environment == "Outdoor": score += 1
    if group_size == "Alone":      score -= 1
    elif group_size == "SmallGroup": score += 1
    elif group_size == "LargeGroup": score += 2

    if score >= 3:   return [0.10, 0.25, 0.65]
    elif score == 2: return [0.15, 0.45, 0.40]
    elif score == 1: return [0.25, 0.55, 0.20]
    elif score == 0: return [0.40, 0.45, 0.15]
    else:            return [0.65, 0.25, 0.10]


def interest_type_probs(social_energy, time_restriction, group_size, budget):
    weights = {k: 1.0 for k in STATES["InterestType"]}

    if social_energy == "High":
        weights["Music"] += 2.0; weights["Sports"] += 2.0
        weights["Networking"] += 2.0; weights["Food"] += 1.0
    elif social_energy == "Medium":
        weights["Food"] += 1.0; weights["Networking"] += 1.0
        weights["Games"] += 1.0; weights["Music"] += 1.0
    elif social_energy == "Low":
        weights["Culture"] += 2.0; weights["Games"] += 2.0
        weights["Food"] += 1.0

    if time_restriction == "Short":
        weights["Networking"] += 1.0; weights["Games"] += 1.0; weights["Food"] += 1.0
    elif time_restriction == "Medium":
        weights["Music"] += 1.0; weights["Food"] += 1.0
        weights["Culture"] += 1.0; weights["Games"] += 1.0
    elif time_restriction == "Long":
        weights["Music"] += 1.0; weights["Sports"] += 1.0
        weights["Culture"] += 1.0; weights["Food"] += 1.0

    if group_size == "Alone":
        weights["Culture"] += 2.0; weights["Games"] += 2.0
    elif group_size == "Pair":
        weights["Food"] += 2.0; weights["Culture"] += 1.0; weights["Games"] += 1.0
    elif group_size == "SmallGroup":
        weights["Games"] += 2.0; weights["Food"] += 1.0; weights["Networking"] += 1.0
    elif group_size == "LargeGroup":
        weights["Music"] += 2.0; weights["Sports"] += 2.0
        weights["Networking"] += 2.0; weights["Food"] += 1.0

    if budget == "Free":
        weights["Networking"] += 1.0; weights["Games"] += 1.0
        weights["Culture"] += 1.0; weights["Food"] += 1.0
    elif budget == "Low":
        weights["Food"] += 1.0; weights["Games"] += 1.0; weights["Culture"] += 1.0
    elif budget == "Medium":
        weights["Music"] += 1.0; weights["Food"] += 1.0; weights["Sports"] += 1.0
    elif budget == "High":
        weights["Music"] += 2.0; weights["Sports"] += 1.0; weights["Culture"] += 1.0

    return normalize([weights[k] for k in STATES["InterestType"]])


def event_score_matching_probs(user_history, interest_type):
    if user_history == "Unknown":
        return [0.20, 0.55, 0.25]
    if user_history == interest_type:
        return [0.05, 0.20, 0.75]
    related_pairs = {
        ("Music", "Culture"), ("Culture", "Music"),
        ("Food", "Games"),    ("Games", "Food"),
        ("Networking", "Culture"), ("Culture", "Networking"),
        ("Sports", "Networking"),  ("Networking", "Sports")
    }
    if (user_history, interest_type) in related_pairs:
        return [0.20, 0.60, 0.20]
    return [0.65, 0.25, 0.10]


# Maps each interest category to its primary event
_INTEREST_TO_EVENT = {
    "Music":      "Concert",
    "Food":       "FoodFestival",
    "Sports":     "SportsEvent",
    "Culture":    "MuseumVisit",
    "Networking": "StudentMeetup",
    "Games":      "BoardGameNight",
}

def recommended_event_probs(interest, score):
    """
    InterestType + EventScoreMatching → P(RecommendedEvent)

    High score + matching interest → strongly recommends that event.
    Low score                      → spreads probability, avoids that event.
    """
    weights = {ev: 1.0 for ev in STATES["RecommendedEvent"]}

    # How much the matching event is boosted / suppressed
    boost_map = {"High": 5.0, "Medium": 2.5, "Low": -0.4}
    boost = boost_map[score]

    target_event = _INTEREST_TO_EVENT.get(interest)
    if target_event:
        weights[target_event] = max(0.1, weights[target_event] + boost)

    # Picnic is a mild fallback for food + outdoor vibe
    if interest == "Food" and score != "Low":
        weights["Picnic"] += 0.5

    return normalize([weights[ev] for ev in STATES["RecommendedEvent"]])


#  BUILD THE NETWORK
def build_event_recommendation_bn():
    bn = gum.BayesNet("SocialEventRecommendationBN")

    for node_name, labels in STATES.items():
        add_variable(bn, node_name, labels)

    for parent, child in BN_ARCS:
        bn.addArc(parent, child)

    # Root node priors
    bn.cpt("RecognizedUser").fillWith([0.75, 0.25])

    bn.cpt("EnvironmentPreference").fillWith([0.45, 0.35, 0.20])

    bn.cpt("GroupSize").fillWith([0.10, 0.20, 0.45, 0.25])

    bn.cpt("BudgetPreference").fillWith([0.25, 0.35, 0.30, 0.10])

    # UserPreferenceHistory (depends on RecognizedUser) 
    bn.cpt("UserPreferenceHistory")[{"RecognizedUser": "Known"}] = [
        0.16, 0.16, 0.12, 0.16, 0.18, 0.17, 0.05
    ]
    bn.cpt("UserPreferenceHistory")[{"RecognizedUser": "Unknown"}] = [
        0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.58
    ]

    # Mood depends on UserPreferenceHistory
    mood_cpts = {
        "Music":      [0.25, 0.65, 0.10],
        "Food":       [0.45, 0.45, 0.10],
        "Sports":     [0.15, 0.75, 0.10],
        "Culture":    [0.70, 0.20, 0.10],
        "Networking": [0.25, 0.60, 0.15],
        "Games":      [0.65, 0.25, 0.10],
        "Unknown":    [0.40, 0.35, 0.25],
    }
    for history, probs in mood_cpts.items():
        bn.cpt("Mood")[{"UserPreferenceHistory": history}] = probs

    # SocialEnergy
    for mood, env, grp in itertools.product(
        STATES["Mood"], STATES["EnvironmentPreference"], STATES["GroupSize"]
    ):
        bn.cpt("SocialEnergy")[
            {"Mood": mood, "EnvironmentPreference": env, "GroupSize": grp}
        ] = social_energy_probs(mood, env, grp)

    # TimeRestrictions
    for env, grp in itertools.product(
        STATES["EnvironmentPreference"], STATES["GroupSize"]
    ):
        bn.cpt("TimeRestrictions")[
            {"EnvironmentPreference": env, "GroupSize": grp}
        ] = time_restriction_probs(env, grp)

    # InterestType
    for se, tr, grp, bud in itertools.product(
        STATES["SocialEnergy"], STATES["TimeRestrictions"],
        STATES["GroupSize"],    STATES["BudgetPreference"]
    ):
        bn.cpt("InterestType")[
            {"SocialEnergy": se, "TimeRestrictions": tr,
             "GroupSize": grp, "BudgetPreference": bud}
        ] = interest_type_probs(se, tr, grp, bud)

    # EventScoreMatching
    for hist, it in itertools.product(
        STATES["UserPreferenceHistory"], STATES["InterestType"]
    ):
        bn.cpt("EventScoreMatching")[
            {"UserPreferenceHistory": hist, "InterestType": it}
        ] = event_score_matching_probs(hist, it)

    # RecommendedEvent
    for it, score in itertools.product(
        STATES["InterestType"], STATES["EventScoreMatching"]
    ):
        bn.cpt("RecommendedEvent")[
            {"InterestType": it, "EventScoreMatching": score}
        ] = recommended_event_probs(it, score)

    return bn


#  INFERENCE API
def validate_evidence(evidence: dict) -> None:
    for node, value in evidence.items():
        if node not in STATES:
            raise ValueError(f"Unknown node in evidence: {node}")
        if value not in STATES[node]:
            raise ValueError(
                f"Invalid value '{value}' for node '{node}'. "
                f"Allowed: {STATES[node]}"
            )


def get_posterior_dict(inference, node_name: str) -> dict:
    posterior = inference.posterior(node_name)
    return {
        state: float(posterior[{node_name: state}])
        for state in STATES[node_name]
    }


def recommend_events(evidence: dict) -> tuple[list, dict]:
    """
    Run BN inference given evidence and return:
      - ranking : [(event_name, probability), ...] sorted best-first
      - inferred : posteriors of intermediate nodes for explanation
    """
    validate_evidence(evidence)

    bn        = build_event_recommendation_bn()
    inference = gum.LazyPropagation(bn)
    inference.setEvidence(evidence)
    inference.makeInference()

    posterior = inference.posterior("RecommendedEvent")
    ranking = sorted(
        [(ev, float(posterior[{"RecommendedEvent": ev}]))
         for ev in STATES["RecommendedEvent"]],
        key=lambda x: x[1], reverse=True
    )

    inferred_nodes = {
        "SocialEnergy":      get_posterior_dict(inference, "SocialEnergy"),
        "TimeRestrictions":  get_posterior_dict(inference, "TimeRestrictions"),
        "InterestType":      get_posterior_dict(inference, "InterestType"),
        "EventScoreMatching":get_posterior_dict(inference, "EventScoreMatching"),
    }

    return ranking, inferred_nodes


#  EXPORT HELPERS
def save_bn(bn, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    gum.saveBN(bn, filename)
    print(f"Saved BN → {filename}")


def export_bn_to_dot(bn, dot_filename):
    os.makedirs(os.path.dirname(dot_filename), exist_ok=True)
    dot_code = bn.toDot()
    with open(dot_filename, "w", encoding="utf-8") as f:
        f.write(dot_code)
    print(f"Saved DOT → {dot_filename}")


def render_dot_to_png(dot_filename, png_filename):
    os.makedirs(os.path.dirname(png_filename), exist_ok=True)
    try:
        subprocess.run(
            ["dot", "-Tpng", dot_filename, "-o", png_filename], check=True
        )
        print(f"Rendered PNG → {png_filename}")
        return True
    except FileNotFoundError:
        print("Graphviz 'dot' not found. Install Graphviz to render PNG.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to render PNG: {e}")
    return False


def save_demo_bn_outputs(bn, output_dir=DEMO_OUTPUT_DIR):
    """Save BN demo artifacts into demo_output."""
    os.makedirs(output_dir, exist_ok=True)

    bn_file = os.path.join(output_dir, "event_recommendation_bn.bif")
    dot_file = os.path.join(output_dir, "event_recommendation_bn.dot")
    png_file = os.path.join(output_dir, "event_recommendation_bn.png")

    save_bn(bn, bn_file)
    export_bn_to_dot(bn, dot_file)
    render_dot_to_png(dot_file, png_file)

    return {
        "bn": bn_file,
        "dot": dot_file,
        "png": png_file,
    }


# Standalone test / demo output generator
if __name__ == "__main__":
    print("=== EventMate BN Standalone Demo ===\n")

    test_evidence = {
        "RecognizedUser":        "Known",
        "UserPreferenceHistory": "Games",
        "Mood":                  "Relaxed",
        "EnvironmentPreference": "Indoor",
        "GroupSize":             "SmallGroup",
        "BudgetPreference":      "Low"
    }

    bn = build_event_recommendation_bn()
    output_files = save_demo_bn_outputs(bn)

    print("\nDemo evidence:")
    for node, value in test_evidence.items():
        print(f"  {node:<24} = {value}")

    ranking, inferred = recommend_events(test_evidence)

    print("\nRanked Event Recommendations:")
    for rank, (event, prob) in enumerate(ranking, 1):
        bar = "█" * int(prob * 40)
        print(f"  {rank}. {event:<20} {prob:.3f}  {bar}")

    print("\nInferred Intermediate Nodes:")
    for node, dist in inferred.items():
        top = max(dist, key=dist.get)
        print(f"  {node}: {top}  (p={dist[top]:.3f})")

    print("\nDemo output files:")
    for label, path in output_files.items():
        print(f"  {label.upper():<3} → {path}")
