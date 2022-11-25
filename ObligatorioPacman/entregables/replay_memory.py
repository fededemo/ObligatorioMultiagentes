import random
from collections import namedtuple
from typing import List

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done', 'next_state', 'view_distance'))


# Ejemplo uso
# nueva_tupla = Transition(state, action, reward, done, next_state, view_distance)
class ReplayMemory:
    buffer_size: int
    memory: List
    position: int

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.memory = []
        self.position = 0

    def add(self, state, action, reward, done, next_state, view_distance):
        nueva_tupla = Transition(state, action, reward, done, next_state, view_distance)

        if self.position == self.buffer_size:
            self.position = 0

        if len(self.memory) < self.buffer_size:
            self.memory.append(nueva_tupla)
        else:
            self.memory[self.position] = nueva_tupla

        self.position += 1

    def sample(self, batch_size):
        return random.choices(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)
