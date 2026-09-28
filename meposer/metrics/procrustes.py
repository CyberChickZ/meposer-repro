# Procrustes alignment copied from UnrealEgo (https://github.com/hiroyasuakada/UnrealEgo, utils/util.py),
# the image-only baseline the EMHI paper compares against; used for PA-MPJPE.
import numpy as np


def compute_similarity_transform(S1, S2):
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    var1 = np.sum(X1**2)

    K = X1.dot(X2.T)

    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    R = V.dot(Z.dot(U.T))

    scale = np.trace(R.dot(K)) / var1

    t = mu2 - scale*(R.dot(mu1))

    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def pa_mpjpe(pred, gt):
    errors = []
    for p, g in zip(pred, gt):
        aligned = compute_similarity_transform(p, g)
        errors.append(np.sqrt(np.sum((g - aligned) ** 2, axis=1)).mean())
    return float(np.mean(errors))
