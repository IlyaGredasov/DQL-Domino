# Domino Agent AI

This is my pet project for reinforcement learning study.

### Gist

I've learnt AI that can play simple domino.
You can run neuro-agent, model game for it and ask help for best move.

### Math behind

AI is based on double deep-Q learning algorithm (DQL).
$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \Big[ r_{t+1} + \gamma \cdot Q_{\text{target}}(s_{t+1}, \arg\max_a Q_{\text{policy}}(s_{t+1}, a)) - Q(s_t, a_t) \Big]$

Where:

| Symbol       | Meaning                   |
|--------------|---------------------------|
| $s_t$        | current state             |
| $a_t$        | action taken              |
| $r_{t+1}$    | reward received           |
| $\gamma$     | discount factor (0.99)    |
| $\alpha$     | learning rate             |
| `policy_net` | selects next action       |
| `target_net` | evaluates selected action |

---

### How to play
- Clone this repo via ```git clone https://github.com/IlyaGredasov/DQL-Domino```
- Make venv ```python -m venv .venv```
- Activate venv (depends on OS)
- Load libraries ```pip install -r requirements.txt```
- run ```python play_with_me.py```
- Follow next instructions
