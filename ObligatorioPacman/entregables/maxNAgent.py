import random

import numpy as np

from game_logic import game_util, mcts_util
from game_logic.game import Agent
from game_logic.gameExtended import GameStateExtended


class MaxNAgent(Agent):
    def __init__(self, index, max_depth=2, unroll_type="MC", max_unroll_depth=5, number_of_unrolls=10,
                 view_distance=(2, 2)):
        super().__init__(index)
        self.max_depth = max_depth
        self.unroll_type = unroll_type
        self.max_unroll_depth = max_unroll_depth
        self.number_of_unrolls = number_of_unrolls
        self.view_distance = view_distance

    def evaluationFunction(self, gameState: GameStateExtended, agentIndex):
        processed_obs = game_util.process_state(gameState, self.view_distance, agentIndex)
        # TODO: Implementar función de evaluación que utilice "processed_obs"
        return np.zeros(gameState.getNumAgents())

    def getAction(self, gameState):
        action, value = self.maxN(gameState, self.index, self.max_depth)
        return action

    def maxN(self, gameState: GameStateExtended, agentIndex, depth):
        # TODO: Implementar

        # Casos base:
        if depth == 0:
            pass
        elif gameState.isEnd():
            pass

        # Llamada recursiva
        legalActions = gameState.getLegalActions(agentIndex)
        random.shuffle(legalActions)
        nextAgent = self.getNextAgentIndex(agentIndex, gameState)

        # nextStatesValues = ?

        best_action = None
        best_score_array = np.zeros(gameState.getNumAgents())

        return best_action, best_score_array

    def getNextAgentIndex(self, agentIndex, gameState):
        # TODO: Implementar función
        return 0

    def random_unroll(self, gameState: GameStateExtended, agentIndex):
        # TODO: Implementar función
        return np.zeros(gameState.getNumAgents())

    def montecarlo_eval(self, gameState, agentIndex):
        # TODO: Implementar función
        # Pista: usar random_unroll
        return np.zeros(gameState.getNumAgents())

    def montecarlo_tree_search_eval(self, gameState, agentIndex):
        # TODO: Implementar función
        # PISTA: Utilizar selection_stage, expansion_stage, random_unroll y back_prop_stage
        root = mcts_util.MCTSNode(parent=None, action=None, player=agentIndex, numberOfAgents=gameState.getNumAgents())
        state = gameState.deepCopy()
        for _ in range(self.number_of_unrolls):
            pass

        return np.zeros(gameState.getNumAgents())

    def selection_stage(self, node, gameState):
        # TODO: Implementar función
        return node, gameState

    def expansion_stage(self, node, gameState):
        # TODO: Implementar función
        return node, gameState

    def back_prop_stage(self, node, value):
        # TODO: Implementar función
        pass
