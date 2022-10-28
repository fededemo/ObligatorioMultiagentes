from game_logic.ghostAgents import RandomGhost
from entregables.maxNAgent import MaxNAgent
from game_logic.randomPacman import RandomPacman
from game_logic.PacmanEnvAbs import PacmanEnvAbs
import random
import math
import numpy as np
from game_logic.game_util import process_state

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

def get_default_agents(starting_index, num_ghosts = 10):
    agents = []
    for i in range(starting_index, starting_index + num_ghosts):
        agents.append(RandomGhost(index = i))
    return agents

def run_one_layout(layout = "mediumGrid"):   
    pacman_agent = RandomPacman(index = 0)
    ghost_agent_0 = RandomGhost(index=1)
    #ghost_agent_0 = MaxNAgent(index = 1, unroll_type="MCTS", max_unroll_depth=12, number_of_unrolls=6)
    ghost_agent_1 = RandomGhost(index = 2)
    agents = [pacman_agent, ghost_agent_0, ghost_agent_1]
    agents.extend(get_default_agents(3, 10))    
    done = False
    env = PacmanEnvAbs(agents = agents, view_distance = (2, 2))      
    game_state = env.reset(enable_render= True, layout_name= layout)
    turn_index = 0
    while (not(done)):
        view = process_state(game_state, (2,2), turn_index)
        print(view)

        action = agents[turn_index].getAction(game_state)
        game_state, rewards, done, info = env.step(action, turn_index)        
        turn_index = (turn_index + 1) % env._get_num_agents()

    print(layout, "Pacman Won," if info["win"]
            else "Pacman Lose,", "Scores:", game_state.get_rewards())

if __name__ == '__main__':
    run_one_layout("contestClassic")
    #run_one_layout("mediumClassic")