import os
import subprocess
import pyagrum as gum


def add_variable(bn, name, labels):
    """
    Add a discrete variable with custom labels to the Bayesian network.
    """
    var = gum.LabelizedVariable(name, name, 0)

    for label in labels:
        var.addLabel(label)

    return bn.add(var)


def build_event_recommendation_bn():
    """
    Bayesian Network for preference-based social event recommendation.

    Design:
        RecommendedEvent is the hidden target variable.
        User preferences are observed as evidence.
        The BN infers which event best explains the user's preferences.
    """

    bn = gum.BayesNet("SocialEventRecommendationBN")

    # Target node
    add_variable(
        bn,
        "RecommendedEvent",
        [
            "Concert",
            "FoodFestival",
            "SportsEvent",
            "MuseumNight",
            "StudentMeetup",
            "BoardGameNight",
            "Picnic"
        ]
    )

    # Preference / context nodes
    add_variable(bn, "Mood", ["Relaxed", "Excited", "Stressed"])
    add_variable(bn, "SocialEnergy", ["Low", "Medium", "High"])
    add_variable(bn, "BudgetPreference", ["Free", "Low", "Medium", "High"])
    add_variable(bn, "TimeAvailability", ["Short", "Medium", "Long"])
    add_variable(bn, "IndoorOutdoorPreference", ["Indoor", "Outdoor", "NoPreference"])
    add_variable(bn, "GroupSize", ["Alone", "Pair", "SmallGroup", "LargeGroup"])
    add_variable(bn, "InterestType", ["Music", "Food", "Sports", "Culture", "Networking", "Games"])

    # Bayesian network structure
    # RecommendedEvent explains the observed user preferences
    for node in [
        "Mood",
        "SocialEnergy",
        "BudgetPreference",
        "TimeAvailability",
        "IndoorOutdoorPreference",
        "GroupSize",
        "InterestType"
    ]:
        bn.addArc("RecommendedEvent", node)

    # Prior probability of events
    bn.cpt("RecommendedEvent").fillWith([
        0.15,  # Concert
        0.15,  # FoodFestival
        0.12,  # SportsEvent
        0.14,  # MuseumNight
        0.14,  # StudentMeetup
        0.15,  # BoardGameNight
        0.15   # Picnic
    ])

    # CPT: P(Mood | RecommendedEvent)
    bn.cpt("Mood")[{"RecommendedEvent": "Concert"}] = [0.15, 0.75, 0.10]
    bn.cpt("Mood")[{"RecommendedEvent": "FoodFestival"}] = [0.35, 0.55, 0.10]
    bn.cpt("Mood")[{"RecommendedEvent": "SportsEvent"}] = [0.10, 0.80, 0.10]
    bn.cpt("Mood")[{"RecommendedEvent": "MuseumNight"}] = [0.75, 0.15, 0.10]
    bn.cpt("Mood")[{"RecommendedEvent": "StudentMeetup"}] = [0.30, 0.55, 0.15]
    bn.cpt("Mood")[{"RecommendedEvent": "BoardGameNight"}] = [0.70, 0.20, 0.10]
    bn.cpt("Mood")[{"RecommendedEvent": "Picnic"}] = [0.60, 0.30, 0.10]

    # CPT: P(SocialEnergy | RecommendedEvent)
    bn.cpt("SocialEnergy")[{"RecommendedEvent": "Concert"}] = [0.10, 0.30, 0.60]
    bn.cpt("SocialEnergy")[{"RecommendedEvent": "FoodFestival"}] = [0.20, 0.50, 0.30]
    bn.cpt("SocialEnergy")[{"RecommendedEvent": "SportsEvent"}] = [0.10, 0.30, 0.60]
    bn.cpt("SocialEnergy")[{"RecommendedEvent": "MuseumNight"}] = [0.60, 0.30, 0.10]
    bn.cpt("SocialEnergy")[{"RecommendedEvent": "StudentMeetup"}] = [0.20, 0.45, 0.35]
    bn.cpt("SocialEnergy")[{"RecommendedEvent": "BoardGameNight"}] = [0.55, 0.35, 0.10]
    bn.cpt("SocialEnergy")[{"RecommendedEvent": "Picnic"}] = [0.35, 0.45, 0.20]

    # CPT: P(BudgetPreference | RecommendedEvent)
    bn.cpt("BudgetPreference")[{"RecommendedEvent": "Concert"}] = [0.05, 0.20, 0.55, 0.20]
    bn.cpt("BudgetPreference")[{"RecommendedEvent": "FoodFestival"}] = [0.05, 0.25, 0.55, 0.15]
    bn.cpt("BudgetPreference")[{"RecommendedEvent": "SportsEvent"}] = [0.10, 0.30, 0.45, 0.15]
    bn.cpt("BudgetPreference")[{"RecommendedEvent": "MuseumNight"}] = [0.20, 0.50, 0.25, 0.05]
    bn.cpt("BudgetPreference")[{"RecommendedEvent": "StudentMeetup"}] = [0.45, 0.40, 0.10, 0.05]
    bn.cpt("BudgetPreference")[{"RecommendedEvent": "BoardGameNight"}] = [0.30, 0.55, 0.10, 0.05]
    bn.cpt("BudgetPreference")[{"RecommendedEvent": "Picnic"}] = [0.40, 0.45, 0.10, 0.05]

    # CPT: P(TimeAvailability | RecommendedEvent)
    bn.cpt("TimeAvailability")[{"RecommendedEvent": "Concert"}] = [0.10, 0.30, 0.60]
    bn.cpt("TimeAvailability")[{"RecommendedEvent": "FoodFestival"}] = [0.10, 0.35, 0.55]
    bn.cpt("TimeAvailability")[{"RecommendedEvent": "SportsEvent"}] = [0.15, 0.35, 0.50]
    bn.cpt("TimeAvailability")[{"RecommendedEvent": "MuseumNight"}] = [0.20, 0.60, 0.20]
    bn.cpt("TimeAvailability")[{"RecommendedEvent": "StudentMeetup"}] = [0.45, 0.40, 0.15]
    bn.cpt("TimeAvailability")[{"RecommendedEvent": "BoardGameNight"}] = [0.20, 0.60, 0.20]
    bn.cpt("TimeAvailability")[{"RecommendedEvent": "Picnic"}] = [0.15, 0.45, 0.40]

    # CPT: P(IndoorOutdoorPreference | RecommendedEvent)
    bn.cpt("IndoorOutdoorPreference")[{"RecommendedEvent": "Concert"}] = [0.60, 0.25, 0.15]
    bn.cpt("IndoorOutdoorPreference")[{"RecommendedEvent": "FoodFestival"}] = [0.20, 0.65, 0.15]
    bn.cpt("IndoorOutdoorPreference")[{"RecommendedEvent": "SportsEvent"}] = [0.35, 0.50, 0.15]
    bn.cpt("IndoorOutdoorPreference")[{"RecommendedEvent": "MuseumNight"}] = [0.85, 0.05, 0.10]
    bn.cpt("IndoorOutdoorPreference")[{"RecommendedEvent": "StudentMeetup"}] = [0.75, 0.10, 0.15]
    bn.cpt("IndoorOutdoorPreference")[{"RecommendedEvent": "BoardGameNight"}] = [0.90, 0.03, 0.07]
    bn.cpt("IndoorOutdoorPreference")[{"RecommendedEvent": "Picnic"}] = [0.05, 0.90, 0.05]

    # CPT: P(GroupSize | RecommendedEvent)
    bn.cpt("GroupSize")[{"RecommendedEvent": "Concert"}] = [0.10, 0.25, 0.45, 0.20]
    bn.cpt("GroupSize")[{"RecommendedEvent": "FoodFestival"}] = [0.10, 0.25, 0.45, 0.20]
    bn.cpt("GroupSize")[{"RecommendedEvent": "SportsEvent"}] = [0.10, 0.20, 0.40, 0.30]
    bn.cpt("GroupSize")[{"RecommendedEvent": "MuseumNight"}] = [0.25, 0.35, 0.30, 0.10]
    bn.cpt("GroupSize")[{"RecommendedEvent": "StudentMeetup"}] = [0.20, 0.25, 0.35, 0.20]
    bn.cpt("GroupSize")[{"RecommendedEvent": "BoardGameNight"}] = [0.05, 0.20, 0.60, 0.15]
    bn.cpt("GroupSize")[{"RecommendedEvent": "Picnic"}] = [0.10, 0.25, 0.45, 0.20]

    # CPT: P(InterestType | RecommendedEvent)
    bn.cpt("InterestType")[{"RecommendedEvent": "Concert"}] = [0.80, 0.05, 0.03, 0.05, 0.05, 0.02]
    bn.cpt("InterestType")[{"RecommendedEvent": "FoodFestival"}] = [0.05, 0.80, 0.03, 0.05, 0.05, 0.02]
    bn.cpt("InterestType")[{"RecommendedEvent": "SportsEvent"}] = [0.03, 0.05, 0.80, 0.03, 0.05, 0.04]
    bn.cpt("InterestType")[{"RecommendedEvent": "MuseumNight"}] = [0.05, 0.05, 0.02, 0.80, 0.05, 0.03]
    bn.cpt("InterestType")[{"RecommendedEvent": "StudentMeetup"}] = [0.05, 0.05, 0.05, 0.10, 0.70, 0.05]
    bn.cpt("InterestType")[{"RecommendedEvent": "BoardGameNight"}] = [0.03, 0.05, 0.03, 0.05, 0.04, 0.80]
    bn.cpt("InterestType")[{"RecommendedEvent": "Picnic"}] = [0.10, 0.25, 0.10, 0.10, 0.10, 0.35]

    return bn


def recommend_events(evidence):
    """
    Run inference and return ranked event recommendations.

    Example evidence:
        {
            "Mood": "Relaxed",
            "SocialEnergy": "Low",
            "BudgetPreference": "Low",
            "TimeAvailability": "Medium",
            "IndoorOutdoorPreference": "Indoor",
            "GroupSize": "SmallGroup",
            "InterestType": "Games"
        }
    """

    bn = build_event_recommendation_bn()

    inference = gum.LazyPropagation(bn)
    inference.setEvidence(evidence)
    inference.makeInference()

    posterior = inference.posterior("RecommendedEvent")

    event_labels = [
        "Concert",
        "FoodFestival",
        "SportsEvent",
        "MuseumNight",
        "StudentMeetup",
        "BoardGameNight",
        "Picnic"
    ]

    ranking = []

    for event in event_labels:
        probability = float(posterior[{"RecommendedEvent": event}])
        ranking.append((event, probability))

    ranking.sort(key=lambda x: x[1], reverse=True)

    return ranking


def save_bn(bn, filename):
    """Save the BN in a supported format (PKL, BIF, XDSL, etc.)."""
    gum.saveBN(bn, filename)
    print(f"Saved Bayesian network to {filename}")


def export_bn_to_dot(bn, dot_filename):
    """Export the BN structure to a DOT file."""
    dot_code = bn.toDot()
    with open(dot_filename, "w", encoding="utf-8") as f:
        f.write(dot_code)
    print(f"Saved DOT graph to {dot_filename}")


def render_dot_to_png(dot_filename, png_filename):
    """Use Graphviz dot to render the DOT file into a PNG image."""
    try:
        subprocess.run(["dot", "-Tpng", dot_filename, "-o", png_filename], check=True)
        print(f"Rendered PNG graph to {png_filename}")
    except FileNotFoundError:
        print("Graphviz 'dot' command not found. Install Graphviz to render PNG.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to render PNG from DOT: {e}")


def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


if __name__ == "__main__":
    output_dir = ensure_output_dir(os.path.join(os.path.dirname(__file__), "output"))
    bn = build_event_recommendation_bn()

    save_bn(bn, os.path.join(output_dir, "event_recommendation_bn.pkl"))
    export_bn_to_dot(bn, os.path.join(output_dir, "event_recommendation_bn.dot"))
    render_dot_to_png(os.path.join(output_dir, "event_recommendation_bn.dot"), os.path.join(output_dir, "event_recommendation_bn.png"))

    user_evidence = {
        "Mood": "Relaxed",
        "SocialEnergy": "Low",
        "BudgetPreference": "Low",
        "TimeAvailability": "Medium",
        "IndoorOutdoorPreference": "Indoor",
        "GroupSize": "SmallGroup",
        "InterestType": "Games"
    }

    recommendations = recommend_events(user_evidence)

    print("Ranked Event Recommendations:")
    for event, prob in recommendations:
        print(f"{event}: {prob:.3f}")