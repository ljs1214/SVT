import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


k = []
def svt_solve(A,Omega,tau=None,delta = None, epsilon = 1e-2, max_iterations = 1000):
    Y = np.zeros_like(A)

    if not tau:
        tau = 5 * np.sum(A.shape) / 2
    if not delta:
        delta = 1.2 * np.prod(A.shape) / np.sum(Omega)

    for _ in range(max_iterations):
        U, S, V = np.linalg.svd(Y, full_matrices = False)
        S = np.maximum(S - tau, 0)
        X = np.linalg.multi_dot([U,np.diag(S),V])
        Y += delta*Omega*(A-X)
        rel_recon_error = np.linalg.norm(Omega*(X-A)) / np.linalg.norm(Omega*A)
        k.append(rel_recon_error)
        if rel_recon_error < epsilon:
            break
    print(rel_recon_error)
    return X

def input_mat(name):
    data = scio.loadmat(str(name)+".mat")
    return np.array(data[name]).T



a = input_mat("netsa")
b = input_mat("netss")
c = np.hstack((a,b))
print(c)



data_test = c
data_test = data_test.astype(float)
shape = data_test.shape
Omega2 = np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        if data_test[i,j]>0:
            Omega2[i,j] = 1
print(Omega2)
print(svt_solve(data_test,Omega2))
plt.plot(k)
plt.show()