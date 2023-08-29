"""
List of one factorisations for the first five regular graphs.
"""

k2 = [[(0, 1)]]

k3 = [[(0, 1)],
      [(0, 2)],
      [(1, 2)]]

k4 = [[(0, 1), (2, 3)],
      [(0, 2), (1, 3)],
      [(0, 3), (1, 2)]]

k5 = [[(0, 1), (2, 3)],
      [(0, 2), (1, 4)],
      [(0, 3), (2, 4)],
      [(0, 4), (1, 3)],
      [(1, 2), (3, 4)]]

k6 = [[(0, 1), (2, 3), (4, 5)],
      [(0, 2), (1, 4), (3, 5)],
      [(0, 3), (1, 5), (2, 4)],
      [(0, 4), (1, 3), (2, 5)],
      [(0, 5), (1, 2), (3, 4)]]


def k(n):
    if n < 2 or n > 6 or type(n) is not int:
        raise ValueError("n must be integer between 2 and 6 (inclusive).")
    return [k2, k3, k4, k5, k6][n - 2]
