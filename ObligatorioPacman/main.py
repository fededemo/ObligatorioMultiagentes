import torch

import numpy as np

from entregables.double_dqn_agent import DoubleDQNAgent
from entregables.maxNAgent import MaxNAgent
from game_logic import game_util
from game_logic.ghostAgents import RandomGhost
from game_logic.PacmanEnvAbs import PacmanEnvAbs
from game_logic.randomPacman import RandomPacman
from entregables.qlearning import DQN_Model


all_layouts = [
    "custom1",
    "custom2",
    "capsuleClassic",
    "contestClassic",
    "mediumClassic",
    "minimaxClassic",
    "openClassic",
    "originalClassic",
    "smallClassic",
    "testClassic",
    "trappedClassic",
    "trickyClassic",
    "mediumGrid",
    "smallGrid"
]

view_distances = [(2, 2), (4, 4), (6, 6), (8, 8), (10, 10), (15, 15), (20, 20), (30, 30)]

TOTAL_STEPS = 5000000
EPISODES = 10000
STEPS = 100000

EPSILON_INI = 0.02
EPSILON_MIN = 0.01
EPSILON_TIME = (EPSILON_INI - EPSILON_MIN) * TOTAL_STEPS

EPISODE_BLOCK = 10
USE_PRETRAINED = True

BATCH_SIZE = 32
BUFFER_SIZE = 10000

GAMMA = 0.99
LEARNING_RATE = 1e-4

SAVE_BETWEEN_STEPS = 100000

MATRIX_SIZE = 30
ACTION_SPACE_N = 5
AGENT_INDEX = 2
ENV_NAME = 'GhostDQN'

VIEW_DISTANCE = (10,10) 

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(DEVICE)


def get_default_agents(starting_index, num_ghosts=10):
    agents = []
    for i in range(starting_index, starting_index + num_ghosts):
        agents.append(RandomGhost(index=i))
    return agents


def run_one_layout(layout="mediumGrid"):
    pacman_agent = RandomPacman(index=0)
    # ghost_agent_0 = RandomGhost(index=1)
    ghost_agent_0 = MaxNAgent(index=1, unroll_type="MCTS", max_unroll_depth=12, number_of_unrolls=6,
                              view_distance=VIEW_DISTANCE)

    net_a = DQN_Model(input_size=MATRIX_SIZE * MATRIX_SIZE, n_actions=ACTION_SPACE_N).to(DEVICE)
    net_b = DQN_Model(input_size=MATRIX_SIZE * MATRIX_SIZE, n_actions=ACTION_SPACE_N).to(DEVICE)

    def process_state(state, view_distance, agent_index):
        state_view = game_util.process_state(state, view_distance, agent_index)
        state_pad = np.pad(state_view, ((0, MATRIX_SIZE - state_view.shape[0]), (0, MATRIX_SIZE - state_view.shape[1])),
                           'constant', constant_values=1)
        return state_pad

    # ghost_agent_1 = RandomGhost(index=2)
    ghost_agent_1 = DoubleDQNAgent(
        None,  # not used for training
        net_a,
        net_b,
        [],  # not used for training
        AGENT_INDEX,   # not used for training
        [],  # not used for training
        [],  # not used for training
        process_state,
        BUFFER_SIZE,
        BATCH_SIZE,
        LEARNING_RATE,
        GAMMA,
        epsilon_i=EPSILON_INI,
        epsilon_f=EPSILON_MIN,
        epsilon_anneal_time=EPSILON_TIME,
        episode_block=EPISODE_BLOCK,
        use_pretrained=USE_PRETRAINED,
        save_between_steps=SAVE_BETWEEN_STEPS,
        view_distance=VIEW_DISTANCE
    )
    agents = [pacman_agent, ghost_agent_0, ghost_agent_1]
    # agents.extend(get_default_agents(3, 10))
    done = False
    env = PacmanEnvAbs(agents=agents, view_distance=VIEW_DISTANCE)
    game_state = env.reset(enable_render=True, layout_name=layout)
    turn_index = 0
    while not done:
        view = process_state(game_state, VIEW_DISTANCE, turn_index)
        print(view)

        action = agents[turn_index].getAction(game_state)
        game_state, rewards, done, info = env.step(action, turn_index)
        turn_index = (turn_index + 1) % env._get_num_agents()

    print(layout, "Pacman Won," if info["win"] else "Pacman Lose,",
          "Scores:", game_state.get_rewards())


if __name__ == '__main__':
    # run_one_layout("contestClassic")
    # run_one_layout("trickyClassic")
    run_one_layout("originalClassic")
    # run_one_layout("smallGrid")
