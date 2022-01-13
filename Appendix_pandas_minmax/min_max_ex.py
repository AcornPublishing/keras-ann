from sklearn.preprocessing import MinMaxScaler
import numpy as np

x = np.array([
    [-1, 4, 10],
    [0, 4, 8],
    [1, 2, 7]
])

new_scaler = MinMaxScaler()
norm_x = new_scaler.fit_transform(x)
print("norm_x =\n", norm_x)

norm_y = [[0.0, 0.5, 2 / 3]]
print("y =", new_scaler.inverse_transform(norm_y))
