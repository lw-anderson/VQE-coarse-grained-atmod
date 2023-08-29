import numpy as np
from scipy.special import lpmn, sph_harm


def P(n, m, theta):
    """Legendre polynomials"""
    return lpmn(n, m, theta)


def Ymn(m, n, theta):
    """ Spherical harmonics"""
    return sph_harm(m, n, theta, np.pi / 2)


def aniso_ddd(C_para_abc, A, B, C, R_AB, R_BC, R_CA):
    """
    Calculate the anisotropic triple-dipole interaction for three linear molecules, in anisotroopic limit (only
    polarisable in direction out of plane). Following Eqn. (10) from

        S.A.C. McDowell (1996) "On the anisotropy of the triple-dipole dispersion energy for interactions involving
        linear molecules" Mol. Phys. 47 (4) 845-858.

    Notation here follows the definitions in paper and appendix. When calculating W, many terms have been discarded
    since Ymn(θ) = 0 [spherical harmonics] for many m,n when θ = π/2.

    C_para_abc = C_{a,b,c}^{||,||,||} triple-dipole dispersion coefficient in all parallel direction for each molecule.
    A = θ_A: Angle of triangle at oscillator A.
    R_AB: Distance between oscillators A & B.
    Other terms follow from appropriate permutations A <-> B <-> C.
    """

    C9abc = C_para_abc / 9
    prefactor = C9abc * (1 + 3 * np.cos(A) * np.cos(B) * np.cos(C)) / (R_AB ** 3 * R_BC ** 3 * R_CA ** 3)

    # Associated Legendre polynomials
    Pmn_A = lpmn(2, 2, np.cos(A))[0]  # input m,n
    Pmn_B = lpmn(2, 2, np.cos(B))[0]

    def Pm1m2(m1, m2):
        return Pmn_B[m1, 2] * Pmn_A[m2, 2] / (1 + 3 * np.cos(A) * np.cos(B) * np.cos(C))

    def IPm1m2(m1, m2):
        return Pm1m2(m2, m1)

    Q1 = 2 / np.sqrt(6) * Pm1m2(2, 0) - 4 / np.sqrt(6) * Pm1m2(1, 1) + 2 / np.sqrt(6) * Pm1m2(0, 2)
    Q2 = 1 + Pm1m2(1, 1) + 8 * Pm1m2(0, 0)
    Q3 = Pm1m2(2, 2) - 2 * Pm1m2(1, 1) + 4 * Pm1m2(0, 0)
    Q4 = Pm1m2(2, 1) - 2 * Pm1m2(0, 1) - 4 * Pm1m2(0, 0)
    Q5 = 4 / np.sqrt(6) * Pm1m2(2, 1) - 1 / np.sqrt(6) * Pm1m2(1, 2) - 4 / np.sqrt(6) * Pm1m2(0, 1) - 14 / np.sqrt(
        6) * Pm1m2(1, 0)
    Q6 = 2 * Pm1m2(0, 2) - 4 * Pm1m2(2, 0) + Pm1m2(1, 1)
    Q7 = 4 / np.sqrt(6) * Pm1m2(2, 1) - 1 / np.sqrt(6) * Pm1m2(1, 2) + 17 * np.sqrt(6) / 3 * Pm1m2(1, 0) - 8 * np.sqrt(
        6) / 3 * Pm1m2(0, 1)
    Q8 = 1 / np.sqrt(6) * Pm1m2(2, 1) + np.sqrt(6) / 3 * Pm1m2(1, 2) - 8 / np.sqrt(6) * Pm1m2(0, 1) - 10 / np.sqrt(
        6) * Pm1m2(0, 1)

    IQ1 = 2 / np.sqrt(6) * IPm1m2(2, 0) - 4 / np.sqrt(6) * IPm1m2(1, 1) + 2 / np.sqrt(6) * IPm1m2(0, 2)
    IQ2 = 1 + IPm1m2(1, 1) + 8 * IPm1m2(0, 0)
    IQ3 = IPm1m2(2, 2) - 2 * IPm1m2(1, 1) + 4 * IPm1m2(0, 0)
    IQ4 = IPm1m2(2, 1) - 2 * IPm1m2(0, 1) - 4 * IPm1m2(0, 0)
    IQ5 = 4 / np.sqrt(6) * IPm1m2(2, 1) - 1 / np.sqrt(6) * IPm1m2(1, 2) - 4 / np.sqrt(6) * IPm1m2(0, 1) - 14 / np.sqrt(
        6) * IPm1m2(1, 0)
    IQ6 = 2 * IPm1m2(0, 2) - 4 * IPm1m2(2, 0) + IPm1m2(1, 1)
    IQ7 = 4 / np.sqrt(6) * IPm1m2(2, 1) - 1 / np.sqrt(6) * IPm1m2(1, 2) + 17 * np.sqrt(6) / 3 * IPm1m2(1,
                                                                                                       0) - 8 * np.sqrt(
        6) / 3 * IPm1m2(0, 1)
    IQ8 = 1 / np.sqrt(6) * IPm1m2(2, 1) + np.sqrt(6) / 3 * IPm1m2(1, 2) - 8 / np.sqrt(6) * IPm1m2(0, 1) - 10 / np.sqrt(
        6) * IPm1m2(0, 1)

    a = np.pi / 2
    b = np.pi / 2
    c = np.pi / 2

    WaAB = np.sqrt(4 * np.pi / 5) * (
            Ymn(2, 2, a) * (Q1 + np.sqrt(6) * Pm1m2(1, 1))
            + Ymn(0, 2, a) * (Q2 - 3 * Pm1m2(1, 1)))

    IWaAB = np.sqrt(4 * np.pi / 5) * (
            (-1) ** 2 * Ymn(2, 2, b) * (IQ1 + np.sqrt(6) * IPm1m2(1, 1))
            + (-1) ** 0 * Ymn(0, 2, b) * (IQ2 - 3 * IPm1m2(1, 1)))

    WbAB = IWaAB

    WcAB = np.sqrt(4 * np.pi / 5) * (
            Ymn(2, 2, c) * Q1
            + Ymn(0, 2, c) * Q2)

    WabAB = (4 * np.pi / 5) * (
            Ymn(-2, 2, a) * Ymn(2, 2, b) * Q3
            + Ymn(0, 2, a) * Ymn(0, 2, b) * (2 + 3 * Pm1m2(1, 1) - Q2)
            + Ymn(0, 2, a) * Ymn(2, 2, b) * (Q1 + np.sqrt(6) * Pm1m2(1, 1))
            + (-1) ** 2 * Ymn(0, 2, b) * Ymn(2, 2, a) * (IQ1 + np.sqrt(6) * IPm1m2(1, 1)))

    WbcAB = (4 * np.pi / 5) * (
            Ymn(2, 2, b) * Ymn(2, 2, c) * Pm1m2(2, 2)
            + Ymn(2, 2, b) * Ymn(0, 2, c) * Q1
            + 4 * Ymn(2, 2, b) * Ymn(-2, 2, c) * Pm1m2(0, 0)
            + Ymn(0, 2, b) * Ymn(2, 2, c) * (Q1 + 2 * np.sqrt(6) * Pm1m2(1, 1))
            + Ymn(0, 2, b) * Ymn(0, 2, c) * (2 - Q2))

    IWbcAB = (4 * np.pi / 5) * (
            (-1) ** 2 * Ymn(2, 2, a) * Ymn(2, 2, c) * IPm1m2(2, 2)
            + (-1) ** 2 * Ymn(2, 2, a) * Ymn(0, 2, c) * IQ1
            + 4 * (-1) ** 2 * Ymn(2, 2, a) * Ymn(-2, 2, c) * IPm1m2(0, 0)
            + (-1) ** 2 * Ymn(0, 2, a) * Ymn(2, 2, c) * (IQ1 + 2 * np.sqrt(6) * IPm1m2(1, 1))
            + (-1) ** 0 * Ymn(0, 2, a) * Ymn(0, 2, c) * (2 - IQ2))

    WacAB = IWbcAB

    WabcAB = (4 * np.pi / 5) ** (3 / 2) * (
            Ymn(-2, 2, a) * Ymn(2, 2, b) * Ymn(2, 2, c) * 2 * np.sqrt(6) * Pm1m2(2, 0)
            + Ymn(0, 2, a) * Ymn(2, 2, b) * Ymn(2, 2, c) * Pm1m2(2, 2)
            + Ymn(0, 2, a) * Ymn(2, 2, b) * Ymn(0, 2, c) * Q1
            + (-1) ** (-2 + 2 + 2) * Ymn(-2, 2, b) * Ymn(2, 2, a) * Ymn(2, 2, c) * 2 * np.sqrt(6) * IPm1m2(2, 0)
            + (-1) ** (0 + 2 + 2) * Ymn(0, 2, b) * Ymn(2, 2, a) * Ymn(2, 2, c) * IPm1m2(2, 2)
            + (-1) ** (0 + 2 + 0) * Ymn(0, 2, b) * Ymn(2, 2, a) * Ymn(0, 2, c) * IQ1
            + Ymn(-2, 2, a) * Ymn(2, 2, b) * Ymn(0, 2, c) * (Q3 + 6 * Pm1m2(1, 1))
            + Ymn(0, 2, a) * Ymn(0, 2, b) * Ymn(2, 2, c) * (Q1 - 2 * np.sqrt(6) * Pm1m2(1, 1))
            + Ymn(0, 2, a) * Ymn(0, 2, b) * Ymn(0, 2, c) * (3 * Q2 - 2))

    DDD = prefactor * np.real(1 - WaAB - WbAB - WcAB + WabAB + WacAB + WbcAB - WabcAB)

    return DDD
