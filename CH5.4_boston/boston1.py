from sklearn.datasets import load_boston

X, y = load_boston(return_X_y=True)

print("X = ", X)
print("y = ", y)
print("X.shape =", X.shape)
print("y.shape=", y.shape)
