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
netclop [OPTIONS] COMMAND [ARGS]
```

### Options
* `--config` Path to a custom configuration YAML file

### Commands

#### Stream
Performs significance clustering on network modular structure from positions. Saves results and plots.

```
netclop stream POSITIONS_PATH --output OUTPUT_PATH
```
* `POSITIONS_PATH` Path to the positions file
* `--output OUTPUT_PATH` Path to the output file where the node list will be written

#### Plot
Plots a node list.

```
netclop plot NODE_PATH
```
* `NODE_PATH` Path to a node list. Node names must be integer H3 indices.