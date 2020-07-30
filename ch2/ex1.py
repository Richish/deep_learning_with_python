# scalars
import numpy as np

sc = np.array(12)
print(sc, type(sc), sc.shape, sc.ndim)

vec = np.array([1, 2, 3, 4, 5, 5, 5])
print(vec.shape, vec.ndim)
print(vec)

mat = np.array([
    [1, 2, 3, 4],
    [1, 2, 3, 5],
    [12, 3, 4, 5],
    [4, 3, 3, 2]
])

print(mat.shape, mat.ndim)

x = np.array([
    [
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [12, 3, 4, 5],
        [4, 3, 3, 2],
        [4, 3, 3, 2]
    ],
    [
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [12, 3, 4, 5],
        [4, 3, 3, 2],
        [4, 3, 3, 2]
    ]
])
print(x.shape, x.ndim)
print(x)
# normally 0 to 4d but video data maybe 5d.
# 3 dimensions of space, 4th of time/movememts