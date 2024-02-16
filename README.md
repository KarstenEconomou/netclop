# Network clustering operations
**Net**work **cl**ustering **op**erations (netclop) is a command line interface for geophysical fluid transport network construction and associated clustering operations (e.g., community detection, significance clustering).

## Installation
Use [pipx](https://github.com/pypa/pipx) to install and run in an isolated environment.
```
brew install pipx
pipx ensurepath
```

To install:
```
pipx install netclop
```

To upgrade:
```
pipx upgrade netclop
```

## Use
Particle trajectories must be decomposed into initial and final latitude and longitude coordinates and stored in a positions file in the form `initial_latitude,initial_longitude,final_latitude,final_longitude`. Positions are binned with [H3](https://github.com/uber/h3-py). Community detection uses [Infomap](https://github.com/mapequation/infomap).

```
netclop [GLOBAL OPTIONS] COMMAND [ARGS] [OPTIONS]
```

### Global options
* `--config CONFIG_PATH` Path to a custom configuration YAML file

### Commands

#### Partition
Constructs and partitions a network from positions, optionally demarcating significant assignments from statistical noise.

```
netclop partition POSITIONS_PATH [OPTIONS]
```

##### Arguments
* `POSITIONS_PATH` Path to the positions file.

##### Options
* `--output`, `-o` Path to the output file where the node list will be written.
* `--significance-cluster`, `-sc` Perform significance clustering to delineate noise
* `--res` Specifies the H3 grid resolution (range 0-15) used for domain discretization.
* `--markov-time`, `-mt` Tuning parameter of the spatial scale of detected structure."
* `--variable-markov-time`/`--static-markov-time` Allows for dynamic adjustment of Markov time with varying network density.
* `--cooling-rate`, `-cr` Cooling rate of simulated annealing.

#### Plot
Plots a node list.

```
netclop plot NODE_PATH [OPTIONS]
```

##### Arguments
* `NODE_PATH` Path to a node list. Node names must be integer H3 indices.

##### Options
* `--delineate-noise`, `-dn` Delineates noise in plot if "significant" column flag in node list file