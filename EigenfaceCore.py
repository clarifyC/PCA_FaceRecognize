import numpy as np

def EigenfaceCore(T):
    m = T.mean(axis=1).reshape(-1, 1)  # Computing the average face image m = (1/P)*sum(Tj's)    (j = 1 : P)
    # Train_Number = T.shape(1)

    A = T - m  # all centered images

    L = np.matmul(A.T, A)  # L is the surrogate of covariance matrix C=A*A'.
    values, vectors = np.linalg.eig(L)  # Diagonal elements of D are the eigenvalues for both L=A'*A and C=A*A'.

    L_eig_vec = []
    for i in range(len(values)):
        if (values[i] > 1):
            L_eig_vec.append(vectors[:, i])

    L_eig_vec = np.array(L_eig_vec).T
    Eigenfaces = np.matmul(A, L_eig_vec)

    return m, A, Eigenfaces