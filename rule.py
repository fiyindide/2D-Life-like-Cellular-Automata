# import numpy as np
#
#
# class Rule:
#     def __init__(self, mask: int):
#         self.mask = mask
#         # Decode mask into boolean arrays
#         self.birth, self.survive = decode(mask)
#         if mask < 0 or mask >= (1 << 18):
#             raise ValueError("Rule mask must be a valid 18-bit integer")
#
#     def __repr__(self):
#         return format_rule(self.birth, self.survive)
#
#
# def encode(B_set, S_set):
#     """Converts Birth and Survival sets to an 18-bit integer.
#         Returns the integer value of Birth and Survival sets combined.
#     """
#     mask = 0
#     for b in B_set:
#         mask |= (1 << b)
#     for s in S_set:
#         mask |= (1 << (s + 9))  # Survival starts at the 9th bit
#     return mask
# # 000000000 000001000
# # 000000100
#
# def decode(mask):
#     """Converts 18-bit integer back to boolean arrays of length 9."""
#     birth = np.zeros(9, dtype=bool)
#     survive = np.zeros(9, dtype=bool)
#     for k in range(9):
#         if (mask >> k) & 1:
#             birth[k] = True
#         if (mask >> (k + 9)) & 1:
#             survive[k] = True
#     return birth, survive
#
#
# def parse(rule_str: str):
#     """Parses strings like 'B3/S23' into array lists B and S sets."""
#     parts = rule_str.upper().split('/')
#     b_part = parts[0].replace('B', '')
#     s_part = parts[1].replace('S', '')
#
#     b_set = [int(digit) for digit in b_part]
#     s_set = [int(digit) for digit in s_part]
#     return b_set, s_set
#
#
# def format_rule(birth_bool, survive_bool):
#     """Converts boolean arrays back to 'B.../S...' string format."""
#     b_str = "".join(str(i) for i, val in enumerate(birth_bool) if val)
#     s_str = "".join(str(i) for i, val in enumerate(survive_bool) if val)
#     return f"B{b_str}/S{s_str}"
#
#


import numpy as np

class Rule:
    def __init__(self, mask: int):
        if mask < 0 or mask >= (1 << 18):
            raise ValueError("Rule mask must be a valid 18-bit integer")
        self.mask = mask
        # Decode mask into boolean arrays
        self.birth, self.survive = decode(mask)
        # Convert to neighbor-count lists for human readability
        self.b_list = bool_to_list(self.birth)
        self.s_list = bool_to_list(self.survive)

    def __repr__(self):
        return format_rule(self.birth, self.survive)

def bool_to_list(bool_array):
    """Converts a boolean array of length 9 to a list of neighbor counts (0-8)."""
    return [i for i, val in enumerate(bool_array) if val]

def encode(B_set, S_set):
    """Converts Birth and Survival sets to an 18-bit integer."""
    mask = 0
    for b in B_set:
        mask |= (1 << b)
    for s in S_set:
        mask |= (1 << (s + 9))
    return mask

def decode(mask):
    """Converts 18-bit integer back to boolean arrays of length 9."""
    birth = np.zeros(9, dtype=bool)
    survive = np.zeros(9, dtype=bool)
    for k in range(9):
        if (mask >> k) & 1:
            birth[k] = True
        if (mask >> (k + 9)) & 1:
            survive[k] = True
    return birth, survive

def parse(rule_str: str):
    """Parses strings like 'B3/S23' into array lists B and S sets."""
    parts = rule_str.upper().split('/')
    b_part = parts[0].replace('B', '')
    s_part = parts[1].replace('S', '')
    b_set = [int(digit) for digit in b_part]
    s_set = [int(digit) for digit in s_part]
    return b_set, s_set

def format_rule(birth_bool, survive_bool):
    """Converts boolean arrays back to 'B.../S...' string format."""
    b_str = "".join(map(str, bool_to_list(birth_bool)))
    s_str = "".join(map(str, bool_to_list(survive_bool)))
    return f"B{b_str}/S{s_str}"