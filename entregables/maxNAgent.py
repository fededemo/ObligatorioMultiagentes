import random
from typing import Tuple

import numpy as np

from game_logic import game_util, mcts_util
from game_logic.game import Agent
from game_logic.gameExtended import GameStateExtended
from game_logic.util import manhattanDistance


class MaxNAgent(Agent):
    def __init__(self, index: int, max_depth: int = 2, unroll_type: str = "MC", max_unroll_depth: int = 5,
                 number_of_unrolls: int = 10,  view_distance: Tuple[int, int] = (2, 2)):
        super().__init__(index)
        self.max_depth = max_depth
        self.unroll_type = unroll_type
        self.max_unroll_depth = max_unroll_depth
        self.number_of_unrolls = number_of_unrolls
        self.view_distance = view_distance

    def reward_logic(self, gameState: GameStateExtended, agentIndex: int) -> float:
        """
        Function that returns the reward for the agent
        :param gameState: state of the game
        :param agentIndex: agent playing.
        :return: reward for the agent
        """
        reward = 0.0
        processed_obs = game_util.process_state(gameState, self.view_distance, agentIndex)

        pacman_pos = (-1, -1)
        agent_pos = (-1, -1)
        agent_status = -1
        food_count = 0
        capsule_pos = []
        for i in range(len(processed_obs)):
            for j in range(len(processed_obs[0])):
                if processed_obs[i][j] == 6:  # Pacman
                    pacman_pos = (i, j)
                elif processed_obs[i][j] == 7 or processed_obs[i][j] == 8:  # Agent
                    agent_pos = (i, j)
                    agent_status = 1 if processed_obs[i][j] == 8 else 0  # 1 = scared, 0 = normal
                elif processed_obs[i][j] == 2:  # Food
                    food_count += 1
                elif processed_obs[i][j] == 3:  # Capsule
                    capsule_pos.append((i, j))

        if pacman_pos != (-1, -1):
            distance_to_pacman = manhattanDistance(agent_pos, pacman_pos)

            reward_distance = 1 / distance_to_pacman

            if agent_status == 1:  # Agent is scared
                reward_distance = reward_distance * -1

            reward += reward_distance * 10
        else:  # if pacman is not visible
            if agent_status == 1:  # Agent is scared
                reward += 100
            else:
                reward -= 10

        food_reward = food_count / processed_obs.shape[0] * processed_obs.shape[1]
        if agent_status == 1:  # Agent is scared
            food_reward = food_reward * -1

        reward += food_reward

        # if agent_status == 1:  # Agent is scared
        #     capsule_reward = 0
        #     for c_pos in capsule_pos:
        #         capsule_distance = manhattanDistance(agent_pos, c_pos)
        #         capsule_reward += -1 / capsule_distance
        #
        #     reward += capsule_reward * 2

        return reward

    def evaluationFunction(self, gameState: GameStateExtended, agentIndex: int):
        # pensar en ponderar si llega al final del tablero para que las rewards de acá sean más altas
        rewards = np.zeros(gameState.getNumAgents())
        if gameState.isEnd():
            if agentIndex != 0 and gameState._check_ghost_has_eaten_pacman(agentIndex):
                agent_reward = 1000 + gameState.data.scores[0]
            else:
                agent_reward = -10000
        else:
            agent_reward = self.reward_logic(gameState, agentIndex)

        rewards[agentIndex] = agent_reward
        return rewards

    def getAction(self, gameState: GameStateExtended):
        action, value = self.maxN(gameState, self.index, self.max_depth)
        return action

    @staticmethod
    def __nparray(value, size) -> np.array:
        """
        Creates a numpy array of size `size` with all values set to `value`
        :param value: Value to set
        :param size: Size of the array
        :return: a numpy array with the given size and filled with the given value
        """
        empties = np.empty(size)
        empties.fill(value)
        return empties

    def maxN(self, gameState: GameStateExtended, agentIndex: int, depth: int) -> Tuple[Tuple[int, int], np.array]:
        # TODO: Implementar
        # Casos base:
        if gameState.isEnd():
            return None, self.evaluationFunction(gameState, agentIndex)
        elif depth == 0:
            if self.unroll_type == "MC":
                return None, self.montecarlo_eval(gameState, agentIndex)
            else:  # "MCTS"
                return None, self.montecarlo_tree_search_eval(gameState, agentIndex)

        legal_actions = gameState.getLegalActions(agentIndex)
        random.shuffle(legal_actions)
        next_agent = self.getNextAgentIndex(agentIndex, gameState.getNumAgents())

        best_action = None
        best_score_array = MaxNAgent.__nparray(-np.Inf, gameState.getNumAgents())  # np.zeros(gameState.getNumAgents())
        for action in legal_actions:
            next_game_state = gameState.generateSuccessor(agentIndex, action)
            _, score_array = self.maxN(next_game_state, next_agent, depth - 1)

            if best_score_array[agentIndex] < score_array[agentIndex]:
                best_action = action
                best_score_array = score_array

        return best_action, best_score_array

    def getNextAgentIndex(self, agentIndex, number_of_agents):
        return (agentIndex + 1) % number_of_agents

    def random_unroll(self, gameState: GameStateExtended, agentIndex) -> np.array:
        # TODO: Implementar función

        if gameState.isEnd():
            return self.evaluationFunction(gameState, agentIndex)

        legal_actions = gameState.getLegalActions(agentIndex)
        random.shuffle(legal_actions)
        action_to_take = legal_actions[0]
        next_game_state = gameState.generateSuccessor(agentIndex, action_to_take)
        next_agent = self.getNextAgentIndex(agentIndex, gameState.getNumAgents())

        if next_game_state.isEnd():
            return self.evaluationFunction(next_game_state, agentIndex)

        for d in range(self.max_unroll_depth):
            legal_actions = next_game_state.getLegalActions(next_agent)
            random.shuffle(legal_actions)
            action_to_take = legal_actions[0]
            next_game_state = next_game_state.generateSuccessor(next_agent, action_to_take)
            next_agent = self.getNextAgentIndex(next_agent, next_game_state.getNumAgents())
            if next_game_state.isEnd():
                return self.evaluationFunction(next_game_state, agentIndex)

        # acá retornar una evaluacion estimada
        return self.evaluationFunction(next_game_state, agentIndex)

    def montecarlo_eval(self, gameState: GameStateExtended, agentIndex: int) -> np.array:
        # TODO: Implementar función
        # Pista: usar random_unroll
        return self.random_unroll(gameState, agentIndex)

    def montecarlo_tree_search_eval(self, gameState: GameStateExtended, agentIndex: int):
        # TODO: Implementar función
        # PISTA: Utilizar selection_stage, expansion_stage, random_unroll y back_prop_stage
        root = mcts_util.MCTSNode(parent=None, action=None, player=agentIndex, numberOfAgents=gameState.getNumAgents())
        state = gameState.deepCopy()

        for _ in range(self.number_of_unrolls):
            # selection stage
            node, gameState = self.selection_stage(root, state)

            # expansion_stage
            node, gameState = self.expansion_stage(node, gameState)

            # random_unroll
            values = self.random_unroll(gameState, agentIndex)

            # backpropagate
            self.back_prop_stage(node, values[agentIndex])

        return root.value

    def selection_stage(self, node: mcts_util.MCTSNode, gameState: GameStateExtended):
        # TODO: Implementar función

        # node.visits += 1 # TODO: Chequear si es acá!!!

        legal_actions = gameState.getLegalActions(node.player)
        random.shuffle(legal_actions)
        for action in legal_actions:
            next_game_state = gameState.generateSuccessor(node.player, action)
            next_agent = self.getNextAgentIndex(node.player, gameState.getNumAgents())
            child = mcts_util.MCTSNode(parent=node, action=action, player=next_agent, numberOfAgents=next_game_state.getNumAgents())
            node.children.append(child)

        if node.explored_children < len(legal_actions):
            selected_node = node.children[node.explored_children]
        else:
            index = np.argmax([child.value[node.player] for child in node.children])
            selected_node = node.children[index]

        selected_state = gameState.generateSuccessor(node.player, selected_node.action)

        node.explored_children += 1
        return selected_node, selected_state

    def expansion_stage(self, node: mcts_util.MCTSNode, gameState: GameStateExtended):
        # TODO: Implementar función
        if gameState.isEnd():
            return node, gameState
        else:
            legal_actions = gameState.getLegalActions(node.player)
            random.shuffle(legal_actions)
            for action in legal_actions:
                next_game_state = gameState.generateSuccessor(node.player, action)
                next_agent = self.getNextAgentIndex(node.player, gameState.getNumAgents())
                new_node = mcts_util.MCTSNode(parent=node, action=action, player=next_agent,
                                              numberOfAgents=next_game_state.getNumAgents())
                node.children.append(new_node)

            return node, gameState

    def back_prop_stage(self, node: mcts_util.MCTSNode, value: float):
        # TODO: Implementar función
        node.visits += 1
        node.value[node.player] += value
        if node.parent is not None:
            self.back_prop_stage(node.parent, node.value[node.player])
