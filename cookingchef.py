from copy import deepcopy
import random
import numpy as np


w = "|"
CHEF = "c"
REWARD = "R"
null = "0"

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

random.seed(42)

n_states = 4
total_episodes = 500

max_steps_episode = 200

minimum_alpha = 0.02

alphas = np.linspace(1.0, minimum_alpha, total_episodes)
gamma = 0.9
eps = 0.2


problem_map = [
    [w, w, w, w, w, w, null, w, w, w, w, w, w],
    [w, null, null, null, null, w, null, w, null, null, null, null, w],
    [w, w, w, null, null, w, null, w, null, w, w, w, w],
    [w, REWARD, w, w, w, w, null, w, null, w, REWARD, null, w],
    [w, null, null, null, null, w, null, w, null, w, null, null, w],
    [w, w, w, w, null, w, null, w, null, w, w, null, w],
    [w, null, null, null, null, w, null, w, null, null, null, null, w],
    [w, w, w, w, w, w, null, w, w, w, w, w, w]
]

for e in problem_map:
    print(' '.join(e))


class State:

    def __init__(self, problem_map, agent_position):
        self.problem_map = problem_map
        self.agent_position = agent_position

    def __eq__(self, other):
        return isinstance(other, State) and self.problem_map == other.problem_map and self.agent_position == other.agent_position

    def __hash__(self):
        return hash(str(self.problem_map) + str(self.agent_position))

    def __str__(self):
        return f"State(grid={self.problem_map}, chef_pos={self.agent_position})"


ACTIONS = [UP, DOWN, LEFT, RIGHT]

initial_state = State(problem_map=problem_map, agent_position=[4, 3])


def act(state, action):
    def new_agent_position(state, action):
        pos = deepcopy(state.agent_position)
        if action == UP:
            pos[0] = max(0, pos[0] - 1)
        elif action == DOWN:
            pos[0] = min(len(state.problem_map) - 1, pos[0] + 1)

        # the gate is on the left
        elif action == LEFT:
            if pos[0] == 6 and pos[1] == 9:
                pos[1] = 5
            else:
                pos[1] = max(0, pos[1] - 1)

        # the gate is on the right
        elif action == RIGHT:
            if pos[0] == 6 and pos[1] == 5:
                pos[1] = 9
            else:
                pos[1] = min(len(state.problem_map[0]) - 1, pos[1] + 1)
        else:
            raise ValueError(f"Invalid action {action}")
        return pos

    p = new_agent_position(state, action)
    grid_item = state.problem_map[p[0]][p[1]]

    new_grid = deepcopy(state.problem_map)

    if grid_item == w:
        reward = -1
        is_complete = True
        new_grid[p[0]][p[1]] += CHEF
    elif grid_item == REWARD:
        reward = 10
        is_complete = True
        new_grid[p[0]][p[1]] += CHEF
    elif grid_item == null:
        reward = 0
        is_complete = False
        old = state.agent_position
        new_grid[old[0]][old[1]] = null
        new_grid[p[0]][p[1]] = CHEF
    elif grid_item == CHEF:
        reward = -0.01
        is_complete = False
    else:
        raise ValueError("Invalid grid item {grid_item}")

    return State(problem_map = new_grid, agent_position = p), reward, is_complete


q_table = dict()


def table(state, action=None):
    if state not in q_table:
        q_table[state] = np.zeros(len(ACTIONS))

    if action is None:
        return q_table[state]

    return q_table[state][action]


def perform_action(state):
    if random.uniform(0, 1) < eps:
        return random.choice(ACTIONS)
    else:
        return np.argmax(table(state))


for i in range(total_episodes):

    state = initial_state
    reward_sum = 0
    alpha = alphas[i]

    for j in range(max_steps_episode):
        action = perform_action(state)
        next_state, reward, done = act(state, action)
        reward_sum += reward

        table(state)[action] = table(state, action) + \
                               alpha * (reward + gamma * np.max(table(next_state)) - table(state, action))
        state = next_state
        if done:
            break
    print(f"No. of Episode {i + 1}: total reward -> {reward_sum}")

r = table(initial_state)
print(f"up={r[UP]}, down={r[DOWN]}, left={r[LEFT]}, right={r[RIGHT]}")

for i, j in q_table.items():
    print("Position(", i.agent_position, ") -> ", j)
