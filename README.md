# Network clustering operations
**Net**work **cl**ustering **op**erations (netclop) is a command line interface for geophysical fluid transport network construction and associated clustering operations (e.g., community detection, significance clustering).

## Installation
Use [pipx](https://github.com/pypa/pipx) to install and run in an isolated environment.
```
brew install pipx
pipx ensurepath
```

```
pipx install netclop
```

## Functions
### Construct network
Particle trajectories must be decomposed into initial and final latitude and longitude coordinates in the form `initial latitude,initial longitude,final latitude,final longitude`. Positions are binned with [h3](https://github.com/uber/h3-py) with specified `-res`.

```
netclop construct coords.csv -o network.csv -res [RES]
```

### Partition network

Weighted, directed networks are represented as an edge list `source node,target node,weight,normalized weight` where the normalized edge weight is such that outgoing edges from each node sum to unity (or zero). Clustering is done using [Infomap](https://github.com/mapequation/infomap) with `-n` outer-loop trials, `-mt` Markov time, and `-s` random seed.

```
netclop partition network.csv -o modules.csv -n [NUM TRIALS] -mt [MARKOV TIME] -s [SEED]
```

### Plot modular structure

Modular descriptions of networks are a node list `node,module,node metric 1,node metric 2,...`. They can be plotted simply with
```
netclop plot modules.csv
```