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

Run `netclop --help` to see all available global options and commands and `netclop COMMAND --help` for the arguments and options of a command.