from dataclasses import dataclass, field
from typing import List
import numpy as np
import random


@dataclass
class ActivationIn:
    source: int = field(default_factory=lambda: random.randint(0, 100))
    amount: float = 0
    weight: float = field(default_factory=lambda: random.random())


@dataclass
class Work:
    activated: List[ActivationIn] = field(default_factory=lambda: [])
    feedback: List[float] = field(default_factory=lambda: [])


class Node:
    def __init__(self, decay_speed, init_num_connections):
        self.activations_in = [ActivationIn() for _ in range(init_num_connections)]
        self.feedback = 0
        self.decay_speed = decay_speed
        self.prob_new = 0.001
        self.trim_thresh = 0.01
        self.feedback_thresh = 0.001

    def step(self, learning_rate=0.5):
        work = Work()

        # - check if it's been activated
        #     - sum([activation_amount * weight for input in activation_in])
        #     - if it has (>1) then we need to add that work to all of the other nodes
        if sum([x.amount for x in self.activations_in]) > 1:
            work.activated = self.connections_out

        # The backprop can happen in the same timestep or seperately from the forward prop
        # - check if there's feedback to propigate
        #     - if there is, then we need to take that feedback level,
        #         - adjust all of the input weights accordingly (smaller if negative, larger if positive)
        #             - weight *= tanh(signal * lr * activation_amount) + 1
        #         - add the backprop work to the nodes so that it backprops
        #             - (address, (signal * activation_amount * ?weight?))
        #     clears backprop signal
        if self.feedback > self.feedback_thresh:
            for activation in self.activations_in:
                activation.weight *= (
                    np.tanh(learning_rate * self.feedback * activation.amount) + 1
                )
                work.feedback.append(self.feedback * activation.amount)
        self.feedback = 0

        # decay's the activations
        #     - [activation_in_node.amount *= self.decay_speed for activation_in_node in activation_in]
        for activation in self.activations_in:
            activation.amount *= self.decay_speed

        # makes new connections
        #     - if rand < prob_new:
        #         - activation_in += (random_address, random_weight)
        if random.random() < self.prob_new:
            self.activations_in += ActivationIn()

        # trims weak connections
        #     - activation_in = [activation for activation in activation_in if activation.weigth > 0.01]
        self.activations_in = [
            activation
            for activation in self.activations_in
            if activation.weight > self.trim_thresh
        ]

        return work


if __name__ == "__main__":
    node = Node(decay_speed=0.01, init_num_connections=10)
    print(node)
    for x in node.activations_in:
        x.amount = random.random()
        print(x)
    node.feedback = -1
    work = node.step()
    print(work)
