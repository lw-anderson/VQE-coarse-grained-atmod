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

"""The base interface for Aqua's gradient."""

from typing import Union, List, Optional

import numpy as np
from aqua import AquaError
from aqua.operators import PauliExpectation
from aqua.operators.gradients.gradient_base import GradientBase
from aqua.operators.list_ops.composed_op import ComposedOp
from aqua.operators.list_ops.list_op import ListOp
from aqua.operators.list_ops.summed_op import SummedOp
from aqua.operators.list_ops.tensored_op import TensoredOp
from aqua.operators.operator_base import OperatorBase
from aqua.operators.operator_globals import Zero, One
from aqua.operators.state_fns.circuit_state_fn import CircuitStateFn
from qiskit.circuit import ParameterExpression, ParameterVector

try:
    from jax import grad, jit
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


class Gradient(GradientBase):
    """Convert an operator expression to the first-order gradient."""

    def convert(self,
                operator: OperatorBase,
                params: Optional[Union[ParameterVector, ParameterExpression,
                                       List[ParameterExpression]]] = None
                ) -> OperatorBase:
        r"""
        Args:
            operator: The operator we are taking the gradient of.
            params: params: The parameters we are taking the gradient with respect to.

        Returns:
            An operator whose evaluation yields the Gradient.

        Raises:
            ValueError: If ``params`` contains a parameter not present in ``operator``.
        """

        if params is None:
            raise ValueError("No parameters were provided to differentiate")

        if isinstance(params, (ParameterVector, list)):
            param_grads = [self.convert(operator, param) for param in params]
            absent_params = [params[i]
                             for i, grad_ops in enumerate(param_grads) if grad_ops is None]
            if len(absent_params) > 0:
                raise ValueError(
                    "The following parameters do not appear in the provided operator: ",
                    absent_params
                )
            return ListOp(param_grads)

        param = params
        # Preprocessing
        expec_op = PauliExpectation(group_paulis=False).convert(operator).reduce()
        cleaned_op = self._factor_coeffs_out_of_composed_op(expec_op)
        return self.get_gradient(cleaned_op, param)

    # pylint: disable=too-many-return-statements
    def get_gradient(self,
                     operator: OperatorBase,
                     params: Union[ParameterExpression, ParameterVector, List[ParameterExpression]]
                     ) -> OperatorBase:
        """Get the gradient for the given operator w.r.t. the given parameters

        Args:
            operator: Operator w.r.t. which we take the gradient.
            params: Parameters w.r.t. which we compute the gradient.

        Returns:
            Operator which represents the gradient w.r.t. the given params.

        Raises:
            ValueError: If ``params`` contains a parameter not present in ``operator``.
            AquaError: If the coefficient of the operator could not be reduced to 1.
            AquaError: If the differentiation of a combo_fn requires JAX but the package is not
                       installed.
            TypeError: If the operator does not include a StateFn given by a quantum circuit
            Exception: Unintended code is reached
        """

        def is_coeff_c(coeff, c):
            if isinstance(coeff, ParameterExpression):
                expr = coeff._symbol_expr
                return expr == c
            return coeff == c

        if isinstance(params, (ParameterVector, list)):
            param_grads = [self.get_gradient(operator, param) for param in params]
            # If get_gradient returns None, then the corresponding parameter was probably not
            # present in the operator. This needs to be looked at more carefully as other things can
            # probably trigger a return of None.
            absent_params = [params[i]
                             for i, grad_ops in enumerate(param_grads) if grad_ops is None]
            if len(absent_params) > 0:
                raise ValueError(
                    'The following parameters do not appear in the provided operator: ',
                    absent_params
                )
            return ListOp(param_grads)

        # By now params is a single parameter
        param = params
        # Handle Product Rules
        if not is_coeff_c(operator._coeff, 1.0):
            # Separate the operator from the coefficient
            coeff = operator._coeff
            op = operator / coeff
            # Get derivative of the operator (recursively)
            d_op = self.get_gradient(op, param)
            # ..get derivative of the coeff
            d_coeff = self.parameter_expression_grad(coeff, param)

            grad_op = 0
            if d_op != ~Zero @ One and not is_coeff_c(coeff, 0.0):
                grad_op += coeff * d_op
            if op != ~Zero @ One and not is_coeff_c(d_coeff, 0.0):
                grad_op += d_coeff * op
            if grad_op == 0:
                grad_op = ~Zero @ One
            return grad_op

        # Base Case, you've hit a ComposedOp!
        # Prior to execution, the composite operator was standardized and coefficients were
        # collected. Any operator measurements were converted to Pauli-Z measurements and rotation
        # circuits were applied. Additionally, all coefficients within ComposedOps were collected
        # and moved out front.
        if isinstance(operator, ComposedOp):

            # Gradient of an expectation value
            if not is_coeff_c(operator._coeff, 1.0):
                raise AquaError('Operator pre-processing failed. Coefficients were not properly '
                                'collected inside the ComposedOp.')

            # Do some checks to make sure operator is sensible
            # TODO add compatibility with sum of circuit state fns
            if not isinstance(operator[-1], CircuitStateFn):
                raise TypeError(
                    'The gradient framework is compatible with states that are given as '
                    'CircuitStateFn')

            return self.grad_method.convert(operator, param)

        elif isinstance(operator, CircuitStateFn):
            # Gradient of an a state's sampling probabilities
            if not is_coeff_c(operator._coeff, 1.0):
                raise AquaError('Operator pre-processing failed. Coefficients were not properly '
                                'collected inside the ComposedOp.')
            return self.grad_method.convert(operator, param)

        # Handle the chain rule
        elif isinstance(operator, ListOp):
            grad_ops = [self.get_gradient(op, param) for op in operator.oplist]

            # Note: this check to see if the ListOp has a default combo_fn
            # will fail if the user manually specifies the default combo_fn.
            # I.e operator = ListOp([...], combo_fn=lambda x:x) will not pass this check and
            # later on jax will try to differentiate it and raise an error.
            # An alternative is to check the byte code of the operator's combo_fn against the
            # default one.
            if operator._combo_fn == ListOp([])._combo_fn:
                return ListOp(oplist=grad_ops)
            elif isinstance(operator, SummedOp):
                return SummedOp(oplist=[grad for grad in grad_ops if grad != ~Zero @ One]).reduce()
            elif isinstance(operator, TensoredOp):
                return TensoredOp(oplist=grad_ops)

            if operator.grad_combo_fn:
                grad_combo_fn = operator.grad_combo_fn
            else:
                if _HAS_JAX:
                    grad_combo_fn = jit(grad(operator._combo_fn, holomorphic=True))
                else:
                    raise AquaError(
                        'This automatic differentiation function is based on JAX. Please install '
                        'jax and use `import jax.numpy as jnp` instead of `import numpy as np` when'
                        'defining a combo_fn.')

            # f(g_1(x), g_2(x)) --> df/dx = df/dg_1 dg_1/dx + df/dg_2 dg_2/dx
            return ListOp([ListOp(operator.oplist, combo_fn=grad_combo_fn), ListOp(grad_ops)],
                          combo_fn=lambda x: np.dot(x[0], x[1]))
