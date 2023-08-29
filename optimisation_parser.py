import argparse


class OptimisationParser(argparse.ArgumentParser):

    def __init__(self):
        super().__init__(description='Execute optimisation of chosen cost function.')

        self.add_argument('--num-oscillators', default=2, type=int, help='Number of 1D drude oscillators.')
        self.add_argument('--encoding', default='bin', type=str, choices=['bin', 'gray'],
                          help='Method for mapping Fock states to qubits.')
        self.add_argument('--gammas', default='[[1.0]]', type=str,
                          help='Lists of coupling constants to be used. Shape should be (num_datapoints, N^2) where N '
                               'is number of oscillators. When num_datapoints > 1, output parameters will be used to '
                               'bootstrap VQE for following gammas.')
        self.add_argument('--geometry', default=None, type=str, choices=['ring', 'regular_polygon'],
                          help='Will override --gammas to determine coupling dependent on oscillator positions.'
                               'Couplings will be calculated relative to first element of gamma list.')
        self.add_argument('--shots', default=1000, type=int, help='Total shots cost function evaluation (shared'
                                                                  ' amongst different terms within Hamiltonian.')
        self.add_argument('--solver', default='aqgd', choices=['midaco', 'aqgd', 'aqgd-fin-diff', 'adam', 'cobyla',
                                                               'nlopt'], type=str)
        self.add_argument('--ansatz', default='vatan', type=str)
        self.add_argument('-n', '--num-qubits', default=4, type=int, help='Number of qubits.')
        self.add_argument('-d', '--depth', default=3, type=int, help='Depth of ansatz.')
        self.add_argument('--cubic', default=0.0, type=float, help='Prefactor for cubic anharmonic potential term.')
        self.add_argument('--quartic', default=0.0, type=float, help='Prefactor for quartic anharmonic potential term.')
        self.add_argument('--ext-field', default=0.0, type=float, help='Value for uniform external value.')
        self.add_argument('--varied-param', default="gammas", type=str, choices=["gammas", "ext-field"],
                          help="Which Hamiltonian parameter is varied for spectrum.")
        self.add_argument('--backend', default='qasm_simulator', type=str, help='Name of IBMQ backend')
        self.add_argument('--noise', default=None, type=str, help='Name of IBMQ device for noise profile.')
        self.add_argument('--extrap-method', default='cnots', type=str, choices=['cnots', 'fold'])
        self.add_argument('--dynamic-shots', action='store_true', help='Allow dynamic shot distribution.')
        self.add_argument('--maxeval', default=100, type=int, help='Maximum number of optimisation steps.')
        self.add_argument('--meas-cal', action='store_true', help='Flag to use measurement calibration.')

        # arguments primarily for MIDACO
        self.add_argument('--cores', default=1, type=int, help='Nummber of cores to use for MIDACO optimiser.')
        self.add_argument('--algostop', default=999999999, type=int, help='For MIDACO optimiser.')
        self.add_argument('--focus', default=1, type=int, help='For MIDACO optimiser.')
        self.add_argument('--ants', default=0, type=int, help='For MIDACO optimiser.')
        self.add_argument('--kernel', default=0, type=int, help='For MIDACO optimiser.')

        # arguments for AQGD
        self.add_argument('--eta', default=0.1, type=float,
                          help='For AQGD, The coefficient of the gradient update. Increasing this value results in '
                               'larger step sizes: param = previous_param - eta * deriv')
        self.add_argument('--momentum', default=0.25, type=float,
                          help='Bias towards the previous gradient momentum in current update. Must be within the '
                               'bounds: [0,1)')

        self.add_argument('--restart', default='', type=str, help='Location of previous outputs to restart.')

        self.add_argument('--read_outputs', action='store_true', help='To perform read_outputs after optimisation.')
        self.add_argument('--save_location', default='output', type=str, help='Subdirectory to save outputs.')
