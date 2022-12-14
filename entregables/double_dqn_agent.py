from os.path import join
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from .abstract_agent import Agent


def to_tensor(elements: np.array) -> torch.Tensor:
    """
    Transforma los elements en un tensor de floats
    :param elements: numpy array de elementos.
    :returns: un tensor cargado con los elementos dados.
    """
    return torch.tensor(elements, dtype=torch.float32)


class DoubleDQNAgent(Agent):
    q_a: nn.Module
    q_b: nn.Module
    loss_function: nn.Module
    optimizer_A: torch.optim.Optimizer
    optimizer_B: torch.optim.Optimizer

    def __init__(self, gym_env: object, model_a: nn.Module, model_b: nn.Module,
                 agents: List[object], agent_idx: int,
                 game_layouts: List[str], view_distances: Tuple[int, int],
                 obs_processing_func: Callable,
                 memory_buffer_size: int, batch_size: int, learning_rate: float,
                 gamma: float, epsilon_i: float, epsilon_f: float, epsilon_anneal_time: int, episode_block: int,
                 use_pretrained: Optional[bool] = False, model_weights_dir_path: Optional[str] = './weights',
                 save_between_steps: Optional[int] = None,
                 view_distance=(2, 2)):
        super().__init__(gym_env, agents, agent_idx, game_layouts, view_distances,
                         obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma,
                         epsilon_i, epsilon_f, epsilon_anneal_time, episode_block,
                         use_pretrained=use_pretrained, model_weights_dir_path=model_weights_dir_path,
                         save_between_steps=save_between_steps,
                         view_distance=view_distance)

        self.model_weights_a_path = join(self.model_weights_dir_path, 'double_DQNAgent_a.pt')
        self.model_weights_b_path = join(self.model_weights_dir_path, 'double_DQNAgent_b.pt')

        # Asignar los modelos al agente (y enviarlos al dispositivo adecuado)
        self.q_a = model_a.to(self.device)
        self.q_b = model_b.to(self.device)

        # Asignar una funci??n de costo (MSE) (y enviarla al dispositivo adecuado)
        self.loss_function = nn.MSELoss().to(self.device)

        # Asignar un optimizador para cada modelo (Adam)
        self.optimizer_A = Adam(self.q_a.parameters(), lr=self.learning_rate)
        self.optimizer_B = Adam(self.q_b.parameters(), lr=self.learning_rate)

        if use_pretrained:
            self._load_net()

    def _predict_action(self, state: np.array, legal_actions: List[int] = []) -> str:
        """
        Dado un estado me predice el action de mayor reward (greedy).
        :param state: state dado.
        :param legal_actions: lista de actions legales.
        :returns: el action con mayor reward dentro de legal actions.
        """
        state_t = to_tensor(state).to(self.device)
        state_t = state_t.unsqueeze(0)

        actions = self.q_a(state_t) + self.q_b(state_t)
        actions = actions.squeeze(0)

        if len(legal_actions) > 0:
            actions = actions[legal_actions]

        action_t = torch.argmax(actions)

        if len(legal_actions) > 0:
            action = legal_actions[action_t.item()]
        else:
            action = action_t.item()

        return self.index_to_action[action]

    def _predict_rewards(self, states: torch.Tensor, use_first: bool = True) -> np.array:
        """
        Dado una serie de estados devuelve las rewards para cada action.
        :param states: states dados.
        :param use_first: to use first network for prediction.
        :returns: la lista de rewards para cada action de cada estado.
        """
        states_t = states.to(self.device)
        if use_first:
            rewards = self.q_a(states_t)
        else:
            rewards = self.q_b(states_t)
        return rewards

    def getAction(self, gameState: object) -> str:
        action = self.select_action(gameState, self.view_distance, train=False)

        # state = self.state_processing_function(gameState, self.view_distance, self.agent_idx)
        # rewards_a = self._predict_rewards(to_tensor([state]), use_first=True)
        # rewards_b = self._predict_rewards(to_tensor([state]), use_first=False)
        # reward = (rewards_a[0][self.action_to_index[action]].item() + rewards_b[0][self.action_to_index[action]].item()) / 2

        return action

    def update_weights(self):
        if len(self.memory) > self.batch_size:
            # Obtener un minibatch de la memoria. Resultando en tensores de estados, acciones, recompensas, flags de
            # terminacion y siguentes estados.
            mini_batch = self.memory.sample(self.batch_size)

            # Enviar los tensores al dispositivo correspondiente.
            states, actions, rewards, dones, next_states, view_distances = zip(*mini_batch)

            states = to_tensor([self.state_processing_function(st, vd, self.agent_idx)
                                for st, vd in zip(states, view_distances)]).to(self.device)
            actions = to_tensor([self.action_to_index[action] for action in actions]).long().to(self.device)
            rewards = to_tensor(rewards).to(self.device)
            dones = to_tensor(dones).to(self.device)
            next_states = to_tensor([self.state_processing_function(st, vd, self.agent_idx)
                                     for st, vd in zip(next_states, view_distances)]).to(self.device)

            # Actualizar al azar Q_a o Q_b usando el otro para calcular el valor de los siguientes estados.
            # Para el Q elegido:
            # Obtener el valor estado-accion (Q) de acuerdo al Q seleccionado.
            use_first = np.random.uniform() > 0.5
            q_actual = self._predict_rewards(states, use_first)
            predicted = q_actual[torch.arange(self.batch_size), actions]

            # Obtener max a' Q para los siguientes estados (del minibatch) (Usando el Q no seleccionado).
            # Es importante hacer .detach() al resultado de este computo.
            # Si el estado siguiente es terminal (done) este valor deber??a ser 0.
            max_q_next_state = torch.max(self._predict_rewards(next_states, not use_first), dim=1).values.detach()

            # Compute el target de DQN de acuerdo a la Ecuacion (3) del paper.
            target = rewards + (1 - dones) * self.gamma * max_q_next_state

            # Resetear gradientes
            if use_first:
                self.optimizer_A.zero_grad()
            else:
                self.optimizer_B.zero_grad()

            # Compute el costo y actualice los pesos.
            loss = self.loss_function(predicted, target)

            loss.backward()

            if use_first:
                torch.nn.utils.clip_grad_norm_(self.q_a.parameters(), max_norm=0.5)
                self.optimizer_A.step()
            else:
                torch.nn.utils.clip_grad_norm_(self.q_b.parameters(), max_norm=0.5)
                self.optimizer_B.step()

    def _save_net(self, suffix: Optional[str] = None) -> None:
        """
        Guarda los pesos de la red a disco.
        :param suffix: sufijo a agregar al archivo.
        """
        file_path_a = self.model_weights_a_path
        file_path_b = self.model_weights_b_path
        if suffix is not None:
            print('INFO: Checkpoint passed, saving partial weights.')
            file_path_a = self.model_weights_a_path.replace('.pt', f'_{suffix}.pt')
            file_path_b = self.model_weights_b_path.replace('.pt', f'_{suffix}.pt')

        torch.save(self.q_a.state_dict(), file_path_a)
        torch.save(self.q_b.state_dict(), file_path_b)

    def _load_net(self) -> None:
        """
        Carga los pesos de la red desde disco.
        """
        print(f"INFO: Using weights from: {self.model_weights_a_path} & {self.model_weights_b_path}")
        if torch.cuda.is_available():
            self.q_a.load_state_dict(torch.load(self.model_weights_a_path))
            # self.q_a.eval()
            self.q_b.load_state_dict(torch.load(self.model_weights_b_path))
            # self.q_b.eval()
        else:
            self.q_a.load_state_dict(torch.load(self.model_weights_a_path, map_location='cpu'))
            self.q_b.load_state_dict(torch.load(self.model_weights_b_path, map_location='cpu'))
