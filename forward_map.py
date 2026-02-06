import numpy as np
from scipy.special import comb
from rule import decode


class AnalyticForwardMap:
    """
        Computes the Forward Map, p(d; B, S)  using the closed form expression for
        calculating the the probability that a cell with 8 neighbors, in a grid
        with density d, has exactly 'k' of those neighbors are alive.
    """
    def __init__(self, d_values):
        self.d = np.array(d_values)
        self.M = len(self.d)

        k_vals = np.arange(9)   # Possible neighbor counts: 0 through 8

        # q_jk = probability of having exactly 'k' alive neighbors for each density
        q = comb(8, k_vals) * (self.d[:, None] ** k_vals) * ((1 - self.d[:, None]) ** (8 - k_vals))

        # Matrix A (M x 9): Prob(Cell is currently DEAD AND has k neighbors)
        self.A = q * (1 - self.d[:, None])

        # Matrix C (M x 9): Prob(Cell is currently ALIVE AND has k neighbors)
        self.C = q * self.d[:, None]

        # Initialize Lookup Tables for all 512 Birth/Survival patterns
        self.PB = np.zeros((512, self.M))
        self.PS = np.zeros((512, self.M))

        for mask in range(512):
            # Convert mask to a bit-array (length 9)
            bits = np.array([(mask >> k) & 1 for k in range(9)])

            # Matrix multiplication to populate the PB lookup table for birth
            self.PB[mask] = self.A @ bits

            # Matrix multiplication to populate the PS lookup table for survival
            self.PS[mask] = self.C @ bits

    def predict(self, rule_mask):
        """
            Returns the predicted next-step density curve for any 18-bit rule mask.
            0x1FF is binary 000000000 111111111 that acts as a filter
            Args:
                rule_mask (int): An integer from 0 to 262,143 representing
                                 the combined Birth and Survival rules.
            Returns:
                np.ndarray: A 1 x M vector of predicted densities (floats).
            """

        # Extract the 9-bit birth and survival masks
        b_mask = rule_mask & 0x1FF  # Lower 9 bits
        s_mask = (rule_mask >> 9) & 0x1FF  # Upper 9 bits

        #   p_pred[j] = p(d_j; B, S)
        p_pred = self.PB[b_mask] + self.PS[s_mask]
        return p_pred