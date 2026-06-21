import os
import subprocess
import itertools
import pyAgrum as gum


STATES = {
    "RecognizedUser": ["Known", "Unknown"],

    "UserPreferenceHistory": [
        "Music",
        "Food",
        "Sports",
        "Culture",
        "Networking",
        "Games",
        "Unknown"
    ],

    "Mood": ["Relaxed", "Excited", "Stressed"],

    "EnvironmentPreference": [
        "Indoor",
        "Outdoor",
        "NoPreference"
    ],

    "GroupSize": [
        "Alone",
        "Pair",
        "SmallGroup",
        "LargeGroup"
    ],

    "BudgetPreference": [
        "Free",
        "Low",
        "Medium",
        "High"
    ],

    # Inference nodes
    "SocialEnergy": ["Low", "Medium", "High"],
    "TimeRestrictions": ["Short", "Medium", "Long"],

    "InterestType": [
        "Music",
        "Food",
        "Sports",
        "Culture",
        "Networking",
        "Games"
    ],

    "EventScoreMatching": ["Low", "Medium", "High"],

    # Output node
    "RecommendedEvent": [
        "Concert",
        "FoodFestival",
        "SportsEvent",
        "MuseumVisit",
        "StudentMeetup",
        "BoardGameNight",
        "Picnic"
    ]
}


BN_ARCS = [
    # From your diagram
    ("RecognizedUser", "UserPreferenceHistory"),

    ("UserPreferenceHistory", "Mood"),
    ("UserPreferenceHistory", "EventScoreMatching"),

    ("Mood", "SocialEnergy"),
    ("EnvironmentPreference", "SocialEnergy"),
    ("GroupSize", "SocialEnergy"),

    ("EnvironmentPreference", "TimeRestrictions"),
    ("GroupSize", "TimeRestrictions"),

    ("SocialEnergy", "InterestType"),
    ("TimeRestrictions", "InterestType"),
    ("GroupSize", "InterestType"),
    ("BudgetPreference", "InterestType"),

    ("InterestType", "EventScoreMatching"),

    ("EventScoreMatching", "RecommendedEvent")
]


def add_variable(bn, name, labels):
    """
    Add a discrete variable with custom labels to the Bayesian network.
    """
    var = gum.LabelizedVariable(name, name, 0)

    for label in labels:
        var.addLabel(label)

    return bn.add(var)


def normalize(values):
    """
    Normalize a list of numbers so they sum to 1.
    """
    total = sum(values)

    if total == 0:
        return [1.0 / len(values)] * len(values)

    return [v / total for v in values]


def social_energy_probs(mood, environment, group_size):
    """
    Inference node:
    Mood + EnvironmentPreference + GroupSize -> SocialEnergy
    """

    score = 0

    if mood == "Excited":
        score += 2
    elif mood == "Relaxed":
        score += 0
    elif mood == "Stressed":
        score -= 2

    if environment == "Outdoor":
        score += 1
    elif environment == "Indoor":
        score += 0
    elif environment == "NoPreference":
        score += 0

    if group_size == "Alone":
        score -= 1
    elif group_size == "Pair":
        score += 0
    elif group_size == "SmallGroup":
        score += 1
    elif group_size == "LargeGroup":
        score += 2

    if score >= 3:
        return [0.05, 0.25, 0.70]  # Low, Medium, High
    elif score == 2:
        return [0.10, 0.40, 0.50]
    elif score == 1:
        return [0.20, 0.55, 0.25]
    elif score == 0:
        return [0.30, 0.50, 0.20]
    elif score == -1:
        return [0.50, 0.40, 0.10]
    else:
        return [0.75, 0.20, 0.05]


def time_restriction_probs(environment, group_size):
    """
    Inference node:
    EnvironmentPreference + GroupSize -> TimeRestrictions
    """

    score = 0

    if environment == "Outdoor":
        score += 1
    elif environment == "Indoor":
        score += 0
    elif environment == "NoPreference":
        score += 0

    if group_size == "Alone":
        score -= 1
    elif group_size == "Pair":
        score += 0
    elif group_size == "SmallGroup":
        score += 1
    elif group_size == "LargeGroup":
        score += 2

    if score >= 3:
        return [0.10, 0.25, 0.65]  # Short, Medium, Long
    elif score == 2:
        return [0.15, 0.45, 0.40]
    elif score == 1:
        return [0.25, 0.55, 0.20]
    elif score == 0:
        return [0.40, 0.45, 0.15]
    else:
        return [0.65, 0.25, 0.10]


def interest_type_probs(social_energy, time_restriction, group_size, budget):
    """
    Inference node:
    SocialEnergy + TimeRestrictions + GroupSize + BudgetPreference -> InterestType
    """

    weights = {
        "Music": 1.0,
        "Food": 1.0,
        "Sports": 1.0,
        "Culture": 1.0,
        "Networking": 1.0,
        "Games": 1.0
    }

    if social_energy == "High":
        weights["Music"] += 2.0
        weights["Sports"] += 2.0
        weights["Networking"] += 2.0
        weights["Food"] += 1.0

    elif social_energy == "Medium":
        weights["Food"] += 1.0
        weights["Networking"] += 1.0
        weights["Games"] += 1.0
        weights["Music"] += 1.0

    elif social_energy == "Low":
        weights["Culture"] += 2.0
        weights["Games"] += 2.0
        weights["Food"] += 1.0

    if time_restriction == "Short":
        weights["Networking"] += 1.0
        weights["Games"] += 1.0
        weights["Food"] += 1.0

    elif time_restriction == "Medium":
        weights["Music"] += 1.0
        weights["Food"] += 1.0
        weights["Culture"] += 1.0
        weights["Games"] += 1.0

    elif time_restriction == "Long":
        weights["Music"] += 1.0
        weights["Sports"] += 1.0
        weights["Culture"] += 1.0
        weights["Food"] += 1.0

    if group_size == "Alone":
        weights["Culture"] += 2.0
        weights["Games"] += 2.0

    elif group_size == "Pair":
        weights["Food"] += 2.0
        weights["Culture"] += 1.0
        weights["Games"] += 1.0

    elif group_size == "SmallGroup":
        weights["Games"] += 2.0
        weights["Food"] += 1.0
        weights["Networking"] += 1.0

    elif group_size == "LargeGroup":
        weights["Music"] += 2.0
        weights["Sports"] += 2.0
        weights["Networking"] += 2.0
        weights["Food"] += 1.0

    if budget == "Free":
        weights["Networking"] += 1.0
        weights["Games"] += 1.0
        weights["Culture"] += 1.0
        weights["Food"] += 1.0

    elif budget == "Low":
        weights["Food"] += 1.0
        weights["Games"] += 1.0
        weights["Culture"] += 1.0

    elif budget == "Medium":
        weights["Music"] += 1.0
        weights["Food"] += 1.0
        weights["Sports"] += 1.0

    elif budget == "High":
        weights["Music"] += 2.0
        weights["Sports"] += 1.0
        weights["Culture"] += 1.0

    return normalize([
        weights["Music"],
        weights["Food"],
        weights["Sports"],
        weights["Culture"],
        weights["Networking"],
        weights["Games"]
    ])


def event_score_matching_probs(user_history, interest_type):
    """
    Inference node:
    UserPreferenceHistory + InterestType -> EventScoreMatching
    """

    if user_history == "Unknown":
        return [0.20, 0.55, 0.25]  # Low, Medium, High

    if user_history == interest_type:
        return [0.05, 0.20, 0.75]

    related_pairs = {
        ("Music", "Culture"),
        ("Culture", "Music"),
        ("Food", "Games"),
        ("Games", "Food"),
        ("Networking", "Culture"),
        ("Culture", "Networking"),
        ("Sports", "Networking"),
        ("Networking", "Sports")
    }

    if (user_history, interest_type) in related_pairs:
        return [0.20, 0.60, 0.20]

    return [0.65, 0.25, 0.10]


def build_event_recommendation_bn():
    """
    Bayesian Network using the user's diagram.

    3-level design:

    Level 1:
        RecognizedUser
        UserPreferenceHistory
        Mood
        EnvironmentPreference
        GroupSize
        BudgetPreference

    Level 2:
        SocialEnergy
        TimeRestrictions
        InterestType
        EventScoreMatching

    Level 3:
        RecommendedEvent
    """

    bn = gum.BayesNet("SocialEventRecommendationBN")

    for node_name, labels in STATES.items():
        add_variable(bn, node_name, labels)

    for parent, child in BN_ARCS:
        bn.addArc(parent, child)

    # Root node: RecognizedUser
    bn.cpt("RecognizedUser").fillWith([
        0.75,  # Known
        0.25   # Unknown
    ])

    # Root node: EnvironmentPreference
    bn.cpt("EnvironmentPreference").fillWith([
        0.45,  # Indoor
        0.35,  # Outdoor
        0.20   # NoPreference
    ])

    # Root node: GroupSize
    bn.cpt("GroupSize").fillWith([
        0.10,  # Alone
        0.20,  # Pair
        0.45,  # SmallGroup
        0.25   # LargeGroup
    ])

    # Root node: BudgetPreference
    bn.cpt("BudgetPreference").fillWith([
        0.25,  # Free
        0.35,  # Low
        0.30,  # Medium
        0.10   # High
    ])

    # UserPreferenceHistory depends on RecognizedUser
    bn.cpt("UserPreferenceHistory")[{"RecognizedUser": "Known"}] = [
        0.16,  # Music
        0.16,  # Food
        0.12,  # Sports
        0.16,  # Culture
        0.18,  # Networking
        0.17,  # Games
        0.05   # Unknown
    ]

    bn.cpt("UserPreferenceHistory")[{"RecognizedUser": "Unknown"}] = [
        0.07,  # Music
        0.07,  # Food
        0.07,  # Sports
        0.07,  # Culture
        0.07,  # Networking
        0.07,  # Games
        0.58   # Unknown
    ]

    # Mood depends on UserPreferenceHistory, as in your diagram
    mood_cpts = {
        "Music": [0.25, 0.65, 0.10],
        "Food": [0.45, 0.45, 0.10],
        "Sports": [0.15, 0.75, 0.10],
        "Culture": [0.70, 0.20, 0.10],
        "Networking": [0.25, 0.60, 0.15],
        "Games": [0.65, 0.25, 0.10],
        "Unknown": [0.40, 0.35, 0.25]
    }

    for history, probs in mood_cpts.items():
        bn.cpt("Mood")[{"UserPreferenceHistory": history}] = probs

    # SocialEnergy inference node
    for mood, environment, group_size in itertools.product(
        STATES["Mood"],
        STATES["EnvironmentPreference"],
        STATES["GroupSize"]
    ):
        bn.cpt("SocialEnergy")[
            {
                "Mood": mood,
                "EnvironmentPreference": environment,
                "GroupSize": group_size
            }
        ] = social_energy_probs(mood, environment, group_size)

    # TimeRestrictions inference node
    for environment, group_size in itertools.product(
        STATES["EnvironmentPreference"],
        STATES["GroupSize"]
    ):
        bn.cpt("TimeRestrictions")[
            {
                "EnvironmentPreference": environment,
                "GroupSize": group_size
            }
        ] = time_restriction_probs(environment, group_size)

    # InterestType inference node
    for social_energy, time_restriction, group_size, budget in itertools.product(
        STATES["SocialEnergy"],
        STATES["TimeRestrictions"],
        STATES["GroupSize"],
        STATES["BudgetPreference"]
    ):
        bn.cpt("InterestType")[
            {
                "SocialEnergy": social_energy,
                "TimeRestrictions": time_restriction,
                "GroupSize": group_size,
                "BudgetPreference": budget
            }
        ] = interest_type_probs(
            social_energy,
            time_restriction,
            group_size,
            budget
        )

    # EventScoreMatching inference node
    for user_history, interest_type in itertools.product(
        STATES["UserPreferenceHistory"],
        STATES["InterestType"]
    ):
        bn.cpt("EventScoreMatching")[
            {
                "UserPreferenceHistory": user_history,
                "InterestType": interest_type
            }
        ] = event_score_matching_probs(user_history, interest_type)

    # RecommendedEvent depends on EventScoreMatching
    bn.cpt("RecommendedEvent")[{"EventScoreMatching": "Low"}] = [
        0.07,  # Concert
        0.10,  # FoodFestival
        0.06,  # SportsEvent
        0.22,  # MuseumVisit
        0.20,  # StudentMeetup
        0.20,  # BoardGameNight
        0.15   # Picnic
    ]

    bn.cpt("RecommendedEvent")[{"EventScoreMatching": "Medium"}] = [
        0.14,  # Concert
        0.15,  # FoodFestival
        0.12,  # SportsEvent
        0.15,  # MuseumVisit
        0.16,  # StudentMeetup
        0.15,  # BoardGameNight
        0.13   # Picnic
    ]

    bn.cpt("RecommendedEvent")[{"EventScoreMatching": "High"}] = [
        0.17,  # Concert
        0.16,  # FoodFestival
        0.13,  # SportsEvent
        0.13,  # MuseumVisit
        0.14,  # StudentMeetup
        0.14,  # BoardGameNight
        0.13   # Picnic
    ]

    return bn


def validate_evidence(evidence):
    """
    Check whether all evidence nodes and values are valid.
    """

    for node, value in evidence.items():
        if node not in STATES:
            raise ValueError(f"Unknown node in evidence: {node}")

        if value not in STATES[node]:
            raise ValueError(
                f"Invalid value '{value}' for node '{node}'. "
                f"Allowed values: {STATES[node]}"
            )


def get_posterior_dict(inference, node_name):
    """
    Convert posterior probability of a node into a Python dictionary.
    """

    posterior = inference.posterior(node_name)

    return {
        state: float(posterior[{node_name: state}])
        for state in STATES[node_name]
    }


def recommend_events(evidence):
    """
    Run inference and return ranked event recommendations.

    Example evidence:
        {
            "RecognizedUser": "Known",
            "UserPreferenceHistory": "Games",
            "Mood": "Relaxed",
            "EnvironmentPreference": "Indoor",
            "GroupSize": "SmallGroup",
            "BudgetPreference": "Low"
        }
    """

    validate_evidence(evidence)

    bn = build_event_recommendation_bn()

    inference = gum.LazyPropagation(bn)
    inference.setEvidence(evidence)
    inference.makeInference()

    posterior = inference.posterior("RecommendedEvent")

    ranking = []

    for event in STATES["RecommendedEvent"]:
        probability = float(posterior[{"RecommendedEvent": event}])
        ranking.append((event, probability))

    ranking.sort(key=lambda x: x[1], reverse=True)

    inferred_nodes = {
        "SocialEnergy": get_posterior_dict(inference, "SocialEnergy"),
        "TimeRestrictions": get_posterior_dict(inference, "TimeRestrictions"),
        "InterestType": get_posterior_dict(inference, "InterestType"),
        "EventScoreMatching": get_posterior_dict(inference, "EventScoreMatching")
    }

    return ranking, inferred_nodes


def save_bn(bn, filename):
    """
    Save the Bayesian Network.
    Recommended extension: .bif
    """
    gum.saveBN(bn, filename)
    print(f"Saved Bayesian network to {filename}")


def export_bn_to_dot(bn, dot_filename):
    """
    Export the BN structure to a DOT file.
    """
    dot_code = bn.toDot()

    with open(dot_filename, "w", encoding="utf-8") as f:
        f.write(dot_code)

    print(f"Saved DOT graph to {dot_filename}")


def export_three_level_dot(dot_filename):
    """
    Export a custom DOT graph with 3 visual levels.
    This keeps the same BN structure but makes the diagram easier to read.
    """

    level_1 = [
        "RecognizedUser",
        "UserPreferenceHistory",
        "Mood",
        "EnvironmentPreference",
        "GroupSize",
        "BudgetPreference"
    ]

    level_2 = [
        "SocialEnergy",
        "TimeRestrictions",
        "InterestType",
        "EventScoreMatching"
    ]

    level_3 = [
        "RecommendedEvent"
    ]

    def label_for(node):
        return node + "\\n{" + ", ".join(STATES[node]) + "}"

    lines = []
    lines.append("digraph SocialEventRecommendationBN {")
    lines.append("  rankdir=TB;")
    lines.append("  node [shape=box, style=rounded];")

    lines.append("  subgraph cluster_level_1 {")
    lines.append('    label="Level 1: Input / Preference Nodes";')
    lines.append("    rank=same;")
    for node in level_1:
        lines.append(f'    {node} [label="{label_for(node)}"];')
    lines.append("  }")

    lines.append("  subgraph cluster_level_2 {")
    lines.append('    label="Level 2: Inference Nodes";')
    lines.append("    rank=same;")
    for node in level_2:
        lines.append(f'    {node} [label="{label_for(node)}"];')
    lines.append("  }")

    lines.append("  subgraph cluster_level_3 {")
    lines.append('    label="Level 3: Output Node";')
    lines.append("    rank=same;")
    for node in level_3:
        lines.append(f'    {node} [label="{label_for(node)}"];')
    lines.append("  }")

    for parent, child in BN_ARCS:
        lines.append(f"  {parent} -> {child};")

    lines.append("}")

    with open(dot_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved 3-level DOT graph to {dot_filename}")


def render_dot_to_png(dot_filename, png_filename):
    """
    Use Graphviz dot to render the DOT file into a PNG image.
    """
    try:
        subprocess.run(
            ["dot", "-Tpng", dot_filename, "-o", png_filename],
            check=True
        )
        print(f"Rendered PNG graph to {png_filename}")

    except FileNotFoundError:
        print("Graphviz 'dot' command not found. Install Graphviz to render PNG.")

    except subprocess.CalledProcessError as e:
        print(f"Failed to render PNG from DOT: {e}")


def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


if __name__ == "__main__":
    output_dir = ensure_output_dir(
        os.path.join(os.path.dirname(__file__), "output")
    )

    bn = build_event_recommendation_bn()

    save_bn(
        bn,
        os.path.join(output_dir, "event_recommendation_bn.bif")
    )

    export_bn_to_dot(
        bn,
        os.path.join(output_dir, "event_recommendation_bn.dot")
    )

    render_dot_to_png(
        os.path.join(output_dir, "event_recommendation_bn.dot"),
        os.path.join(output_dir, "event_recommendation_bn.png")
    )

    export_three_level_dot(
        os.path.join(output_dir, "event_recommendation_bn_3_levels.dot")
    )

    render_dot_to_png(
        os.path.join(output_dir, "event_recommendation_bn_3_levels.dot"),
        os.path.join(output_dir, "event_recommendation_bn_3_levels.png")
    )

    user_evidence = {
        "RecognizedUser": "Known",
        "UserPreferenceHistory": "Games",
        "Mood": "Relaxed",
        "EnvironmentPreference": "Indoor",
        "GroupSize": "SmallGroup",
        "BudgetPreference": "Low"
    }

    recommendations, inferred_nodes = recommend_events(user_evidence)

    print("\nInferred Intermediate Nodes:")
    for node, distribution in inferred_nodes.items():
        print(f"\n{node}:")
        for state, prob in distribution.items():
            print(f"  {state}: {prob:.3f}")

    print("\nRanked Event Recommendations:")
    for event, prob in recommendations:
        print(f"{event}: {prob:.3f}")