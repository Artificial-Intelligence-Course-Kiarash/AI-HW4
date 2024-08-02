import numpy as np

# in here I used forward algorithm for probability of states after observed values
# and I used viterbi algorithm for Most likely sequence of states
initial_state_probs = list(map(float, input().split()))
initial_state_probs = np.array(initial_state_probs)
transition_probs = list()
for i in range(3):
    p = list(map(float, input().split()))
    transition_probs.append(p)
transition_probs = np.array(transition_probs)
emission_probs = list()
for i in range(3):
    p = list(map(float, input().split()))
    emission_probs.append(p)
emission_probs = np.array(emission_probs)
observations = list(map(int, input().split()))
observed = np.array(observations)
T = len(observations)
N = len(initial_state_probs)
viterbi = np.zeros((T, N))
backpointer = np.zeros((T, N), dtype=int)
forward_probs = np.zeros((T, N))
forward_probs[0] = initial_state_probs * emission_probs[:, observations[0]]
for t in range(1, T):
    for s in range(N):
        forward_probs[t, s] = np.sum(forward_probs[t - 1] * transition_probs[:, s]) * emission_probs[s, observations[t]]
state_probs = forward_probs[-1] / np.sum(forward_probs[-1])
viterbi[0] = initial_state_probs * emission_probs[:, observations[0]]
for t in range(1, T):
    for s in range(N):
        viterbi[t, s] = np.max(viterbi[t - 1] * transition_probs[:, s] * emission_probs[s, observations[t]])
        backpointer[t, s] = np.argmax(viterbi[t - 1] * transition_probs[:, s])
best_state = np.argmax(viterbi[T - 1])
state_sequence = [best_state]
for t in range(T - 1, 0, -1):
    best_state = backpointer[t, best_state]
    state_sequence.insert(0, best_state)
for i in range(len(state_sequence)):
    if state_sequence[i] == 0:
        state_sequence[i] = "cold"
    elif state_sequence[i] == 1:
        state_sequence[i] = "normal"
    else:
        state_sequence[i] = "hot"
print("Most likely sequence of states:\n", state_sequence)
for i in range(len(state_probs)):
    state_probs[i] = round(state_probs[i], 3)
print("Probability of states after observing:\n", state_probs)
