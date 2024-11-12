[![PyPI version](https://badge.fury.io/py/netclop.svg)](https://badge.fury.io/py/netclop)
# netclop
**NETwork CLustering OPerations (for geophysical fluid transport).**

`netclop` is a command-line interface for constructing network models of geophysical fluid transport and performing associated clustering operations (e.g., community detection and significance clustering).

![Robust cores of sea scallop connectivity community structure in the Northwest Atlantic](https://github.com/KarstenEconomou/netclop/raw/main/img/geo.png)
![UpSet plot showing core coalescence and stability in the landscape of degenerate community structure](https://github.com/KarstenEconomou/netclop/raw/main/img/upset.png)

## Features
* Binning of Lagrangian particle simulations using [H3](https://github.com/uber/h3)
* Network construction of LPT connectivity
* Community detection using [Infomap](https://github.com/mapequation/infomap)
* Network resampling and recursive significance clustering
* Node centrality calculation
* Spatially-embedded network visualization

## About
`netclop` was created as a CLI to facilitate network-theoretic analysis of marine connectivity in support of larval ecology.
It functions as a library to computations on network ensembles.
Developed at the Department of Engineering Mathematics and Internetworking, Dalhousie University by Karsten N. Economou.

### Papers
* 2024 - [Characterizing variability in complex network community structure with a recursive significance clustering scheme](https://arxiv.org/abs/2409.12852) (Karsten N. Economou, Cassie R. Norman, Wendy C. Gentleman)

## Usage
### CLI
`netclop` accepts Lagrangian particle tracking (LPT) simulations decomposed into initial and final positions in as `.csv` structured as
```
initial_latitude,initial_longitude,final_latitude,final_longitude
```
as an input. Recursive significance clustering is run on all provided filepaths of LPT position files and stores all produced content in the specified output directory
```
netclop rsc [OPTIONS] [PATHS] -o [DIRECTORY]
```
If one LPT position file is given, it will be bootstrapped; otherwise, each LPT position file is treated as an observation.

### Significance clustering
Significance clustering can be run on a `networkx.Graph` object directly, which will partition and bootstrap

```python
from netclop import NetworkEnsemble
ne = NetworkEnsemble(net, **ne_config)
ne.sigclu(**sc_config)
cores = ne.cores
```
or on an ensemble of partitions
```python
from netclop import SigClu
sc = SigClu(partitions, **sc_config)
sc.run()
cores = sc.cores
```