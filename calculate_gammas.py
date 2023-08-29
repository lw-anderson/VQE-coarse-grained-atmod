import numpy as np


def get_arbitrary_gammas(gamma_0, positions):
    """
    Coupling matrix for arbitrary positions of oscillators. Coupling is calculated as gamma_0/D^3 where D is
    separation between oscillator pair. gamma_0 is reference coupling defined for pair at D=1.
    """
    n = len(positions)
    gammas_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                (x1, y1) = positions[i]
                (x2, y2) = positions[j]
                d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                gammas_matrix[i, j] = gamma_0 / (d ** 3)
    return gammas_matrix


def get_polygon_gammas(gamma_0, n):
    """
    Coupling matrix for regular polygon with n vertices.
    """
    theta = 2 * np.pi / n
    r = 0.5  # radius of circle to inscribe polygon
    vertices = [(r * np.cos(i * theta), r * np.sin(i * theta)) for i in range(n)]
    gammas = get_arbitrary_gammas(gamma_0, vertices)
    return gammas


def get_ring_gammas(gamma_0, n):
    """
    Coupling matrix with only nearest neighbour coupling with value gamma_0.
    """
    gammas_matrix = np.zeros((n, n))
    for i in range(n):
        gammas_matrix[i, (i + 1) % n] = gamma_0
        gammas_matrix[i, (i - 1) % n] = gamma_0
    return gammas_matrix


def get_rhombus_gammas(gamma_0, theta):
    """
    Coupling matrix for rhombus defined by angle θ ∈ [0,π/2].
    """
    vertices = [(-np.cos(theta / 2), 0), (0, -np.sin(theta / 2)), (np.cos(theta / 2), 0),
                (0, np.sin(theta / 2))]
    gammas = get_arbitrary_gammas(gamma_0, vertices)

    return gammas


def get_rhombus_with_centre_gammas(gamma_0, theta):
    """
    Coupling matrix for four oscillators in rhombus defined by angle θ ∈ [0,π/2] with additional oscillator placed at
    the centre.
    """
    vertices = [(-np.cos(theta / 2), 0), (0, -np.sin(theta / 2)), (np.cos(theta / 2), 0),
                (0, np.sin(theta / 2)), (0, 0)]
    gammas = get_arbitrary_gammas(gamma_0, vertices)

    return gammas


def get_isoscoles_triangle(gamma_0, phi):
    """
    Coupling matrix for three oscillators in isoscoles triangle defined by angle ϕ ∈ [0,π]. gamma_0 defines coupling
    between pairs along two equal sides of triangle.
    """
    vertices = [(0, 0), (-np.sin(phi / 2), np.cos(phi / 2)), (np.sin(phi / 2), np.cos(phi / 2))]
    return get_arbitrary_gammas(gamma_0, vertices)
