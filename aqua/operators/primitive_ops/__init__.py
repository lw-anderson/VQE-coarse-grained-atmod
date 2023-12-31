# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Primitive Operators (:mod:`aqua.operators.primitive_ops`)
================================================================

.. currentmodule:: aqua.operators.primitive_ops

Operators are defined to be functions which take State functions to State functions.

PrimitiveOps are the classes for representing basic Operators, backed by computational
Operator primitives from Terra. These classes (and inheritors) primarily serve to allow the
underlying primitives to "flow" - i.e. interoperability and adherence to the Operator
formalism - while the core computational logic mostly remains in the underlying primitives.
For example, we would not produce an interface in Terra in which
``QuantumCircuit1 + QuantumCircuit2`` equaled the Operator sum of the circuit
unitaries, rather than simply appending the circuits. However, within the Operator
flow summing the unitaries is the expected behavior.

Note:
     All mathematical methods are not in-place, meaning that they return a
     new object, but the underlying primitives are not copied.

Primitive Operators
===================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   PrimitiveOp
   CircuitOp
   MatrixOp
   PauliOp

"""

from .primitive_op import PrimitiveOp
from .pauli_op import PauliOp
from .matrix_op import MatrixOp
from .circuit_op import CircuitOp

__all__ = ['PrimitiveOp',
           'PauliOp',
           'MatrixOp',
           'CircuitOp']
