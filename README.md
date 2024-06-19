[![PyPI version](https://badge.fury.io/py/netclop.svg)](https://badge.fury.io/py/netclop)
# netclop
**NETwork CLustering OPerations for geophysical fluid transport.**

`netclop` is a command-line interface for constructing network models of geophysical fluid transport and performing associated clustering operations (e.g., community detection and significance clustering).

![Robust cores of sea scallop connectivity community structure in the Northwest Atlantic](https://github.com/KarstenEconomou/netclop/raw/main/img/img.png)

## Features
* Binning of Lagrangian particle simulations using [H3](https://github.com/uber/h3)
* Network construction of fluid transport
* Community detection using [Infomap](https://github.com/mapequation/infomap)
* Network resampling and significance clustering
* Node centrality calculation
* Spatially-embedded network visualization

## About
`netclop` was created to facilitate network-theoretic analysis of marine connectivity in support of larval ecology.
Developed at the Department of Engineering Mathematics and Internetworking, Dalhousie University by Karsten N. Economou.

## Usage
Particle trajectories should be decomposed into initial and final positions in `.csv` form and specified with `--input-data lpt`
```
initial_latitude,initial_longitude,final_latitude,final_longitude
```

Networks are given in the form of a weighted edgelist `.csv` with `--input-data net`
```
source_node,target_node,weight
```
