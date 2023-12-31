# Variational quantum algorithm for electronic coarse-graining

A variational quantum algorithm for finding the ground state of coupled Drude Oscillators used for modelling atomic and
molecular interactions. Circuit simulation and running on real device performed using Qiskit and IBMQ.

The method and results for this work are presented in

[L. W. Anderson, M. Kiffner, P. Kl. Barkoutsos, I. Tavernelli, J. Crain, and D. Jaksch. "Coarse-grained intermolecular interactions on quantum processors." Physical Review A
**105** 062409 (2022).](https://doi.org/10.1103/PhysRevA.105.062409)
([arXiv:2110.00968](https://doi.org/10.48550/arXiv.2110.00968))

## Example running code

### Creating python environment

To create python environment (requires conda):

```
conda env create -f environment.yml
```

To activate:

```
conda activate CoarseGrainedVQE
```

### Running optimisation

Statevector simulator to find the ground state of two coupled oscillators using the AQGD optimiser. Chosen eta = 0.25
and momentum = 0.5 with 100 gradient descent steps.

  ```
  python optimisation.py --backend statevector_simulator --num-oscillators 2 --gammas '[[1.]]' --solver aqgd --eta 0.25 --momentum 0.5 --maxeval 100
  ```

Shot-based QASM simulation with ADAM optimiser.

  ```
  python optimisation.py --backend qasm_simulator --num-oscillators 2 --gammas '[[1.]]' --shots 500 --solver adam --maxeval 100
  ```

### Reading saved outputs

For time intensive simulations or real device experiments, you may want to run the optimisation procedure using an
external cluster or machine and then read and analyse the results locally. Saved outputs can be read using
`read_outputs.py` as follows

  ```
  python read_outputs.py --directory ./output_example/pair_qhos_statevector_simulator
  ```

this generates plots visualising the state and optimisation procedure within the supplied output directory.

The variational optimisation routine is run using `optimisation_run.py`. Runtime arguments define your cost function,
coupling-constant(s) noise model, ansatz, optimisation method, number of measurements and whether you use a
statevector or shot-based simulation. There are additional optional arguments that are specific to the optimisation
method chosen as well arguments that allow you to add non-linear to the Hamiltonian.

## Description of codebase

The main functionality of the repository is used in the `NCoupledQHOFunc` class. This constructs the Hamiltonian,
ansatz circuit and simulator/real device objects in order to evaluate expectation values of the
Hamiltonian using the Qiskit [1] simulators or real IBMQ backends.

When evaluating the cost function using the shot-based simulation, the circuits that need to be evaluated to measure
each term within the Hamiltonian are grouped if the measurements for different terms qubit-wise commute. This grouping
is done using a combination of the
[1-factorisation](https://en.wikipedia.org/wiki/Graph_factorization#Perfect_1-factorization) of the interaction graphs,
as well as the sorted-insertion algorithm [2]. The grouping reduces the number of measurements needed to achieve a
certain shot noise when compared to naively measuring all terms using separate circuits.

For the results in the above paper, we use an ansatz consisting of general SO(4) operations (real valued 2 qubit gates)
described in [3], arranged in a "brick work" layout with nearest-neighbour two qubit interactions. The ansatz depth
can be specified by the user (default depth = 3 works reasonably well for two and three coupled oscillators).
Also include are ansaetze based on Refs. [4,5]

Various gradient and non-gradient based optimisers are also included. This includes
[ADAM](https://qiskit.org/documentation/stubs/qiskit.aqua.components.optimizers.ADAM.html),
[AQGD](https://qiskit.org/documentation/stubs/qiskit.aqua.components.optimizers.AQGD.html),
[COBYLA](https://qiskit.org/documentation/stubs/qiskit.aqua.components.optimizers.COBYLA.html)
and finite difference gradient descent. In the cases of ADAM and finite difference methods, these include modified
versions of Qiskit code. Modified and derivative versions must retain the copyright license included within those files.

## Not contained within the repository

To make this code open access, some functionality that requires paid licenses or non-open source code has been removed
in a minimal way. There may be traces and options remaining for such removed functionality. This includes,
the [MIDACO optimiser](http://www.midaco-solver.com/) as well as methods for performing error mitigation based
on the [Lanczos procedure](https://doi.org/10.22331/q-2021-07-01-492).

## A note on the use of Qiskit-Aqua

This code was originally written in 2019/2020 and made considerable use
of [Qiskit Aqua](https://github.com/qiskit-community/qiskit-aqua). Since then, Qiskit Aqua is no
longer supported (in fact pip/conda installation commonly fails due to cython compilation error). Thus, to include Aqua
functionality, this codebase includes a copy of the Qiskit Aqua code, with minor modifications so that it works with
later versions of qiskit (see folder `./aqua`). Qiskit-aqua uses an Apache 2.0 license.

## References

[1] Qiskit contributors, Qiskit: An Open-source Framework for Quantum Computing (2023),
[doi:10.5281/zenodo.2573505](10.5281/zenodo.2573505)

[2] O. Crawford, et al. "Efficient quantum measurement of Pauli operators in the presence of finite sampling error."
Quantum **5** (2021)

[3] F. Vatan and C. Williams "Optimal quantum circuits for general two-qubit gates" Phys. Rev. A. **69** (2004)

[4] P. Suchsland et al. "Algorithmic error mitigation scheme for current quantum processors." Quantum **5**
(2021)

[5] S. Sim et al. "Expressibility and entangling capability of parameterized quantum circuits for hybrid
quantum‐classical algorithms." Advanced Quantum Technologies **2** (2019)

## License

This code uses an Apache 2.0 license.

