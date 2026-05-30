# Bayesian Network Design for Social Event Recommendation Robot

## 1. Overview

This Bayesian Network is designed for a socially interactive robot that recommends suitable social events to a user based on their preferences. The robot first detects or recognizes the user, greets them, asks preference-related questions, and then uses the Bayesian Network structure to model how different factors influence the final event recommendation.

The purpose of this graph is not only to choose an event, but also to represent uncertainty. For example, the robot may not be fully sure about the user identity, mood, or exact preference. A Bayesian Network is useful because it allows the robot to combine multiple uncertain pieces of information and still make a reasonable recommendation.

---

## 2. Main Idea of the Graph

The graph contains several user preference nodes that influence one central node:

**`event_match_score`**

This node represents how well a potential event matches the user's current preferences.

The final node is:

**`recommended_event`**

This node represents the event category that the robot recommends to the user.

The general flow is:

```text
User information + Preferences
        ↓
Event match score
        ↓
Recommended event
```

---

## 3. Bayesian Network Structure

The graph can be described as:

```text
recognized_user
        ↓
user_preference_history
        ↓
event_match_score
        ↓
recommended_event
```

Other preference nodes also point to `event_match_score`:

```text
mood ────────────────────────────────┐
social_energy ───────────────────────┤
budget_preference ───────────────────┤
time_availability ───────────────────┤
indoor_outdoor_preference ───────────┤──> event_match_score
group_size ──────────────────────────┤
interest_type ───────────────────────┘
```

This means the robot combines the user's mood, budget, time, group size, interest, and location preference to estimate how suitable different social events are.

---

## 4. Node Explanation

### 4.1 `recognized_user`

This node represents whether the robot recognizes the person using face detection or face verification.

Possible states:

```text
known
unknown
```

If the user is known, the robot may use previous preference history. If the user is unknown, the robot should ask more questions instead of making assumptions.

Example robot behavior:

```text
"Hello, nice to see you again!"
```

or

```text
"Hello! I don't think we have met before. I will ask you a few questions."
```

---

### 4.2 `user_preference_history`

This node represents the user's past preferences if the user is recognized.

Possible states:

```text
music
food
sports
culture
networking
games
none
```

For example, if a recognized user usually likes music events, this node may influence the recommendation toward concerts or music-related events.

This node helps with personalization.

---

### 4.3 `mood`

This node represents the user's current mood.

Possible states:

```text
relaxed
excited
stressed
```

The mood can be collected through dialog. For example, the robot may ask:

```text
"Are you looking for something calm or something exciting today?"
```

Mood affects the event choice because a relaxed user may prefer a calm event, while an excited user may prefer a concert, sports event, or festival.

---

### 4.4 `social_energy`

This node represents how socially active the user wants to be.

Possible states:

```text
low
medium
high
```

A user with low social energy may prefer a quiet event, such as a museum night or board game night. A user with high social energy may prefer a concert, student meetup, or sports event.

---

### 4.5 `budget_preference`

This node represents how much the user wants to spend.

Possible states:

```text
free
low
medium
high
```

This helps the robot avoid recommending events that do not match the user's budget. For example, if the user wants a free or low-cost activity, the robot may recommend a picnic, student meetup, museum night, or board game night instead of an expensive concert.

---

### 4.6 `time_availability`

This node represents how much time the user has.

Possible states:

```text
short
medium
long
```

If the user has short time, the robot may recommend a short meetup or quick social event. If the user has more time, longer events such as concerts, food festivals, or sports events become more suitable.

---

### 4.7 `indoor_outdoor_preference`

This node represents whether the user prefers indoor or outdoor activities.

Possible states:

```text
indoor
outdoor
no_preference
```

Examples:

- Indoor: board game night, museum night, student meetup
- Outdoor: picnic, food festival, outdoor sports event
- No preference: any suitable option may be recommended

---

### 4.8 `group_size`

This node represents whether the user is attending alone or with others.

Possible states:

```text
alone
pair
small_group
large_group
```

This matters because some events are better for groups, while others are suitable for individuals. For example, a board game night is good for a small group, while a student meetup may be suitable for both individuals and groups.

---

### 4.9 `interest_type`

This node represents the user's preferred activity category.

Possible states:

```text
music
food
sports
culture
networking
games
```

This is one of the most important preference nodes because it directly represents the user's interest.

Examples:

- `music` may lead to `concert`
- `food` may lead to `food_festival`
- `sports` may lead to `sports_event`
- `culture` may lead to `museum_night`
- `networking` may lead to `student_meetup`
- `games` may lead to `board_game_night`

---

### 4.10 `event_match_score`

This is the central inference node of the graph.

Possible states:

```text
low
medium
high
```

It combines all the evidence from the user preference nodes and estimates how well an event matches the user. A high match score means the event is likely suitable for the user.

This node is useful because the robot can explain its decision:

```text
"I selected this event because it matches your budget, mood, group size, and interest."
```

---

### 4.11 `recommended_event`

This is the final output node.

Possible states:

```text
concert
food_festival
sports_event
museum_night
student_meetup
board_game_night
picnic
```

The robot uses this node to recommend a social event to the user.

Example output:

```text
"Based on your preferences, I recommend a board game night. It is indoor, low-cost, relaxed, and suitable for a small group."
```

---

## 5. How the Robot Uses the Graph

The robot follows these steps:

1. **Detect the user**
   - The robot uses the webcam to detect a face.
   - If face verification is implemented, the robot checks whether the user is known or unknown.

2. **Greet the user**
   - If the user is known, the robot may personalize the greeting.
   - If the user is unknown, the robot uses a general greeting.

3. **Ask preference questions**
   - The robot asks about mood, budget, time availability, group size, interest type, and indoor/outdoor preference.

4. **Update the Bayesian Network**
   - The user's answers are treated as evidence in the Bayesian Network.

5. **Estimate event match**
   - The graph combines all evidence and estimates the event match score.

6. **Recommend an event**
   - The robot selects the most suitable event category.

7. **Explain the recommendation**
   - The robot gives a short reason for the recommendation.

8. **Say goodbye**
   - The robot ends the conversation politely with a farewell gesture.

---

## 6. Example Scenario

Suppose the user says:

```text
I feel relaxed.
I prefer indoor activities.
I have a low budget.
I am going with a small group.
I like games.
```

The robot can map this to the following evidence:

```text
mood = relaxed
indoor_outdoor_preference = indoor
budget_preference = low
group_size = small_group
interest_type = games
```

Based on this evidence, the graph should support a recommendation such as:

```text
recommended_event = board_game_night
```

The robot can then say:

```text
"Based on your preferences, I recommend a board game night. It is relaxed, indoor, low-cost, and suitable for a small group."
```

---

## 7. Why This Graph Is Useful

This Bayesian Network is useful for the project because:

- It connects the dialog system with the recommendation system.
- It supports personalization through the `recognized_user` and `user_preference_history` nodes.
- It can handle uncertainty in user preferences.
- It gives the robot a reasoned way to recommend events.
- It is simple enough to explain and implement for the course project.
- It can be expanded later with more event types or more detailed user preferences.

---

## 8. Suggested pyAgrum Code for Displaying the Graph

```python
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb

# Create Bayesian Network for social event recommendation
bn = gum.fastBN(
    "recognized_user{known|unknown}"
    "->user_preference_history{music|food|sports|culture|networking|games|none};"

    "user_preference_history"
    "->event_match_score{low|medium|high};"

    "mood{relaxed|excited|stressed}"
    "->event_match_score;"

    "social_energy{low|medium|high}"
    "->event_match_score;"

    "budget_preference{free|low|medium|high}"
    "->event_match_score;"

    "time_availability{short|medium|long}"
    "->event_match_score;"

    "indoor_outdoor_preference{indoor|outdoor|no_preference}"
    "->event_match_score;"

    "group_size{alone|pair|small_group|large_group}"
    "->event_match_score;"

    "interest_type{music|food|sports|culture|networking|games}"
    "->event_match_score;"

    "event_match_score"
    "->recommended_event{concert|food_festival|sports_event|museum_night|student_meetup|board_game_night|picnic}"
)

# Show graph
gnb.showBN(bn)
```

---

## 9. Important Note About `{}` Values

In pyAgrum, the values inside `{}` define the possible states of a variable.

For example:

```python
mood{relaxed|excited|stressed}
```

means the node `mood` has three possible states:

```text
relaxed
excited
stressed
```

However, `gnb.showBN(bn)` usually displays only the node names and arrows. It does not show all states inside the graph diagram. The states still exist internally in the Bayesian Network.

To print the states, use:

```python
for name in bn.names():
    variable = bn.variable(name)
    print(name, ":", list(variable.labels()))
```

---

## 10. Conclusion

This Bayesian Network represents the decision-making component of the social event recommendation robot. It receives user information and preferences from the dialog system, combines them through the `event_match_score` node, and outputs a suitable `recommended_event`.

The graph is simple, explainable, and appropriate for a socially interactive robot project because it connects perception, personalization, conversation, recommendation, and explanation.
