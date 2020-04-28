import numpy as np

A = np.array([[3,3,3,4],[5,8,6,5],[7,7,5,7],[8,8,8,8],[9,9,9,9]])


def gradient(A, axis):
    # Get first and last row / col and expand dimensions to match original
    if axis == 0:  # x
        prep = np.expand_dims(A[0, :], axis=0)  # [a,b,c,...] -> [[a,b,c,...]]
        app = np.expand_dims(A[-1, :], axis=0)
    elif axis == 1:  # y
        prep = np.expand_dims(A[:, 0], axis=1)  # [a,b,c,...] -> [[a],[b],[c],...]
        app = np.expand_dims(A[:, -1], axis=1)
    else:
        raise ValueError
    print(prep)
    # Take difference with extra first and last row / col
    D = np.diff(A, axis=axis, prepend=prep, append=app)  # Take difference between adjacent elements
    print(D)

    # Take sum of adjacent rows / col
    if axis == 0:  # x
        B = D[:-1] + D[1:]
    elif axis == 1:  # y
        B = D[:, :-1] + D[:, 1:]
    print(B)

    return B / 2  # Complete formula


print("\n\nFinal:\n", [gradient(A, 0), gradient(A, 1)])

print(np.gradient(A))
