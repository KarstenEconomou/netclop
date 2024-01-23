"""Defines constants."""
type Node = int
type Partition = tuple[set[Node], ...]

DEFAULT_BIN_CONFIG = {
    "res": 5,
}

DEFAULT_PART_CONFIG = {
    "num_trials": 10,
    "markov_time": 2.,
    "seed": 42,
}

DEFAULT_BS_CONFIG = {
    "tuning_param": 1.,
    "size": 1000,
    "seed": 42,
}

DEFAULT_SC_CONFIG = {
    "conf": 0.05,
    "pen_weight": 10.,
    "temp_init": 1.,
    "iter_max": 1000,
    "seed": 42,
    "cool_rate": 0.1,
}
