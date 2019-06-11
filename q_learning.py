from environment import MountainCar
import sys
import numpy as np

mode = sys.argv[1]
weight_out = sys.argv[2]
returns_out = sys.argv[3]
episodes = int(sys.argv[4])
max_iterations = int(sys.argv[5])
epsilon = float(sys.argv[6])
gamma = float(sys.argv[7])
learning_rate = float(sys.argv[8])


class Q:
    def __init__(self, state_space, action_space, mode):
        self.weight = np.zeros((state_space, action_space))
        self.bias = 0.
        self.mode = mode

    def __call__(self, state, action):
        if self.mode == "raw":
            state = np.array([state[0], state[1]])
            return state.dot(self.weight[:, action]) + self.bias
        elif self.mode == "tile":
            state = np.fromiter(state.keys(), dtype=int)
            return np.sum(self.weight[state, action]) + self.bias

    def update(self, state, action, next_state, reward):
        q_true = (reward + gamma * np.max([q(next_state, action) for action in range(3)]))
        q_est = self(state, action)

        if self.mode == "raw":
            state = np.array([state[0], state[1]])
            self.weight[:, action] -= learning_rate * (q_est - q_true) * state
            self.bias -= learning_rate * (q_est - q_true)
        elif self.mode == "tile":
            state = np.fromiter(state.keys(), dtype=int)
            self.weight[state, action] -= learning_rate * (q_est - q_true)
            self.bias -= learning_rate * (q_est - q_true)


if __name__ == "__main__":
    env = MountainCar(mode)
    rewards = []
    q = Q(env.state_space, env.action_space, mode)

    for e in range(episodes):
        state = env.reset()
        summing_reward = 0
        for i in range(max_iterations):
            # choose action
            greedy = np.random.uniform()
            if greedy < epsilon:
                action = np.random.randint(0, 3)
            else:
                qs = [q(state, action) for action in range(3)]
                action = np.argmax(qs)

            # make action
            next_state, reward, done = env.step(action)
            summing_reward += reward

            # update weight
            q.update(state, action, next_state, reward)
            if done:
                break

            state = next_state
        print(summing_reward)
        rewards.append(str(summing_reward))

    with open(weight_out, "w") as f:
        f.write(str(q.bias) + "\n")
        f.writelines("\n".join([str(w) for r in q.weight for w in r]))

    with open(returns_out, "w") as f:
        f.writelines("\n".join(rewards))
