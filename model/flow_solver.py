from __future__ import annotations

from typing import Callable, List, Sequence

import torch
from torch import Tensor


StateList = Sequence[Tensor]
VectorField = Callable[[StateList, StateList, Tensor], StateList]


def _validate(initial_states: StateList, contexts: StateList, num_steps: int) -> None:
    if num_steps not in {1, 2, 4, 6, 8}:
        raise ValueError("num_steps must be one of 1, 2, 4, 6, or 8")
    if not initial_states or len(initial_states) != len(contexts):
        raise ValueError("states and contexts must be non-empty lists of equal length")


def clone_state_list(states: StateList) -> List[Tensor]:
    return [state.clone() for state in states]


def euler_solve(
    initial_states: StateList,
    vector_field_fn: VectorField,
    contexts: StateList,
    num_steps: int = 4,
) -> List[Tensor]:
    _validate(initial_states, contexts, num_steps)
    states = clone_state_list(initial_states)
    dt = 1.0 / num_steps
    for step in range(num_steps):
        time = states[0].new_tensor(step / num_steps)
        velocities = vector_field_fn(states, contexts, time)
        if len(velocities) != len(states):
            raise ValueError("vector field returned the wrong number of scales")
        states = [state + dt * velocity for state, velocity in zip(states, velocities)]
    return states


def solve_flow(
    initial_states: StateList,
    vector_field_fn: VectorField,
    contexts: StateList,
    solver: str = "euler",
    num_steps: int = 4,
) -> List[Tensor]:
    if solver == "euler":
        return euler_solve(initial_states, vector_field_fn, contexts, num_steps)
    raise ValueError("solver must be euler")
