Based on https://github.com/benibienz/TAMER.
This is a library containing popular interactive reinforcement learning algorithms.
- **TAMER** (Training an Agent Manually via Evaluative Reinforcement) is a framework for human-in-the-loop Reinforcement Learning, proposed by [Knox + Stone](http://www.cs.utexas.edu/~sniekum/classes/RLFD-F16/papers/Knox09.pdf) in 2009. 
- **Qlearning** (here the human feedback is treated as an immediate reward).

## How to run
You need python 3.7+ with numpy, sklearn, pygame and gymnasium.

Please create two directories named `logs` and `saved_models`.

You can change the agent type and set credit assignment status in the `config.py` file.

Use `run.py`.

In training, watch the agent play and press 'W' to give a positive reward and 'A' to give a negative. The agent's current action is displayed.
