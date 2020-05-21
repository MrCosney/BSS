import numpy as np


def some_alg1(X, state: dict, options: dict):
    Y = X + 1
    state['some_state_variable1'] = 4.5
    return Y, state


def some_alg2(X, state: dict, options: dict):
    Y = X + 2
    state['some_state_variable2'] = [3, 12]
    return Y, state