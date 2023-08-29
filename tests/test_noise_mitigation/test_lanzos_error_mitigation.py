from unittest import TestCase

from noise_mitigation.lanczos_error_mitigation import lanczos_algorithm


class TestLanczosErrorMitigation(TestCase):

    def test_not_implemented(self):
        self.assertRaises(NotImplementedError, lambda: lanczos_algorithm(None, None, None))
