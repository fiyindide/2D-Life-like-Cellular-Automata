import numpy as np


def neighbor_count_torus(grid):
    """
    Computes the number of alive Moore-neighbors for every cell.
    By using periodic boundary conditions (torus),
    cells on the edge of the grid treat cells on the opposite edge as their
    neighbors. This ensures every cell has exactly 8 neighbor.
    """
    # Shift directions (N,E,W,S,NE,NW,SE,SW)
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),         (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]

    neighbor_count_grid = np.zeros_like(grid, dtype=np.uint8)   # Initalize output grid with zeros

    # Accumulate shifted copies of the grid in each of the 8 Moore directions (row,col)
    for row, col in directions:
        neighbor_count_grid += np.roll(grid, shift=(row, col), axis=(0, 1))

    return neighbor_count_grid


# def test_single_cell():
#     # 1. Create a 3x3 grid with only the center cell (1, 1) alive
#     # Original Grid:
#     # 0 0 0
#     # 0 1 0
#     # 0 0 0
#     grid = np.zeros((3, 3), dtype=np.uint8)
#     grid[1, 1] = 1
#
#     # 2. Compute the neighbor counts
#     neighbor_count_grid = neighbor_count_torus(grid)
#
#     print(neighbor_count_grid)

#test_single_cell()


def step(grid, rule):
    """
    Advances the grid by one time step according to the Rule object.
    """
    k = neighbor_count_torus(grid)

    # compute the next state of the grid by determining where births and survival occurs
    next_state = np.where(grid == 0, rule.birth[k], rule.survive[k])

    return next_state.astype(np.uint8)


def run(grid, rule, T):
    """
    Evolves the grid for T steps and returns the trajectory.
    """
    history = [grid.copy()]
    current = grid
    for t in range(T):
        current = step(current, rule)
        history.append(current.copy())
    return history