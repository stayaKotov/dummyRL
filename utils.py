import numpy as np


def create_obstucles(rows, cols, number):
    res = []
    for _ in range(number):
        lu = np.random.choice(rows-5), np.random.choice(cols-5)
        rd = np.random.choice(range(lu[0]+1, rows)), np.random.choice(range(lu[1]+1, cols))
        res.append([lu, (lu[0], rd[1]), (rd[0], lu[1]), rd])
    return res
