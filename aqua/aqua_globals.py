# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Aqua Globals """

from typing import Optional
import logging

import numpy as np
from qiskit.utils.multiprocessing import local_hardware_info
import qiskit

from .aqua_error import AquaError


logger = logging.getLogger(__name__)


class QiskitAquaGlobals:
    """Aqua class for global properties."""

    CPU_COUNT = local_hardware_info()['cpus']

    def __init__(self) -> None:
        self._random_seed = None  # type: Optional[int]
        self._num_processes = QiskitAquaGlobals.CPU_COUNT
        self._random = None
        self._massive = False

    @property
    def random_seed(self) -> Optional[int]:
        """Return random seed."""
        return self._random_seed

    @random_seed.setter
    def random_seed(self, seed: Optional[int]) -> None:
        """Set random seed."""
        self._random_seed = seed
        self._random = None

    @property
    def num_processes(self) -> int:
        """Return num processes."""
        return self._num_processes

    @num_processes.setter
    def num_processes(self, num_processes: Optional[int]) -> None:
        """Set num processes.
           If 'None' is passed, it resets to QiskitAquaGlobals.CPU_COUNT
        """
        if num_processes is None:
            num_processes = QiskitAquaGlobals.CPU_COUNT
        elif num_processes < 1:
            raise AquaError('Invalid Number of Processes {}.'.format(num_processes))
        elif num_processes > QiskitAquaGlobals.CPU_COUNT:
            raise AquaError('Number of Processes {} cannot be greater than cpu count {}.'
                            .format(num_processes, QiskitAquaGlobals.CPU_COUNT))
        self._num_processes = num_processes
        # TODO: change Terra CPU_COUNT until issue
        # gets resolved: https://github.com/Qiskit/qiskit-terra/issues/1963
        try:
            qiskit.tools.parallel.CPU_COUNT = self.num_processes
        except Exception as ex:  # pylint: disable=broad-except
            logger.warning("Failed to set qiskit.tools.parallel.CPU_COUNT "
                           "to value: '%s': Error: '%s'", self.num_processes, str(ex))

    @property
    def random(self) -> np.random.Generator:
        """Return a numpy np.random.Generator (default_rng)."""
        if self._random is None:
            self._random = np.random.default_rng(self._random_seed)
        return self._random

    @property
    def massive(self) -> bool:
        """Return massive to allow processing of large matrices or vectors."""
        return self._massive

    @massive.setter
    def massive(self, massive: bool) -> None:
        """Set massive to allow processing of large matrices or  vectors."""
        self._massive = massive


# Global instance to be used as the entry point for globals.
aqua_globals = QiskitAquaGlobals()  # pylint: disable=invalid-name
