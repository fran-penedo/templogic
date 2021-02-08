# TempLogic: A Temporal Logic Library

## What is TempLogic?

TempLogic is a library of temporal logics that admit quantitative semantics. It
currently supports Signal Temporal Logic (STL), Tree Spatial Superposition Logic (TSSL)
and Spatial-Temporal Logic (SpaTeL) to varying degrees:

- All three support the construction of every syntactically valid formula and
  computation of quantitative semantics.
- STL has a parser, inference and MILP encoding. 
- TSSL has inference. 
- SpaTeL inference is a work in progress.

## Requirements

You need Python3.8 or newer and the use of virtualenv's or similar is encouraged. The
inference modules require [Weka 3](https://www.cs.waikato.ac.nz/ml/weka/) installed. The MILP
encoding module requires [Gurobi 7](https://www.gurobi.com/) or newer.

## Quickstart

Clone the repository with:

    $ git clone https://github.com/franpenedo/templogic.git

Install with PIP:

    $ pip install templogic
    
If you want to use the inference modules, make sure you have Weka 3 installed, then
run:

    $ pip install templogic[inference]
    
If you want to use the MILP encoding module, make sure you have Gurobi and its Python
package (gurobipy) installed, then run:

    $ pip install templogic[milp]
    
## STL Inference

![Naval Surveillance Scenario](https://franpenedo.com/media/naval.png)
![Naval Surveillance Scenario Result](https://franpenedo.com/media/naval_res.png)

STL inference is the problem of constructing an STL formula that represents "valid" or
"interesting" behavior from samples of "valid" and "invalid" behavior. For example,
suppose you have a set of trajectories of boats approaching a harbor. A subset
of trajectories corresponding with normal behavior as "valid", while the others,
corresponding with behavior consistent with smuggling or terrorist activity, are labeled
"invalid". You can encode this data in a Matlab file with three matrices: 

- A `data` matrix with the trajectories (with shape: number of trajectories x dimensions
  x number of samples), 
- a `t` column vector representing the sampling times, and 
- a `labels` column vector with the labels (in minus-one-plus-one encoding).

You can find the `.mat` file for this example in
`templogic/stlmilp/inference/data/Naval/naval_preproc_data_online.mat`. In order to
obtain an STL formula that represents "valid" behavior, you can run the command:

    $ lltinf learn templogic/stlmilp/inference/data/Naval/naval_preproc_data_online.mat
    
## Publications

A full description of the decision tree approach to STL inference can be found in our
peer-reviewed publication [Bombara, Giuseppe, Cristian-Ioan Vasile, Francisco Penedo,
Hirotoshi Yasuoka, and Calin Belta. “A Decision Tree Approach to Data Classification
Using Signal Temporal Logic.” In Proceedings of the 19th International Conference on
Hybrid Systems: Computation and Control, 1–10. HSCC ’16. New York, NY, USA: ACM, 2016.
https://doi.org/10.1145/2883817.2883843.](https://franpenedo.com/publication/hscc16/).

MILP encoding of STL has been featured in many peer reviewed publications, including our
own work in formal methods for partial differential equations: [Penedo, F., H. Park, and
C. Belta. “Control Synthesis for Partial Differential Equations from Spatio-Temporal
Specifications.” In 2018 IEEE Conference on Decision and Control (CDC), 4890–95, 2018.
https://doi.org/10.1109/CDC.2018.8619313.](https://franpenedo.com/publication/cdc2018/)

Our implementation of TSSL and TSSL inference is based on Bartocci, E., E. Aydin Gol, I.
Haghighi, and C. Belta. “A Formal Methods Approach to Pattern Recognition and Synthesis
in Reaction Diffusion Networks.” IEEE Transactions on Control of Network Systems PP, no.
99 (2016): 1–1. https://doi.org/10.1109/TCNS.2016.2609138.

SpaTeL was first introduced in Haghighi, Iman, Austin Jones, Zhaodan Kong, Ezio
Bartocci, Radu Gros, and Calin Belta. “SpaTeL: A Novel Spatial-Temporal Logic and Its
Applications to Networked Systems.” In Proceedings of the 18th International Conference
on Hybrid Systems: Computation and Control, 189–198. HSCC ’15. New York, NY, USA:
ACM, 2015. https://doi.org/10.1145/2728606.2728633.

## Copyright and Warranty Information

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2016-2021, Francisco Penedo Alvarez (contact@franpenedo.com)
