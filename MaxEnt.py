import numpy as np


class Featrue:
    def __init__(self, value, volume) -> None:
        self.compose = []
        self.value = value


# Generalized Iterative Scaling
def gis(weights, features, bucket_2_w_idx, eps=1e-6):
    delta_z = eps
    delta_z_prev = 10*eps
    while abs(delta_z_prev-delta_z) > eps:
        delta_z_prev = delta_z
        delta_z = 0
        for i, f in enumerate(features):
            sum = 0
            for b in f.compose:
                product = b.volume
                for x in bucket_2_w_idx[b]:
                    product *= weights[x]
                sum += product
            sum /= weights[i]
            z_prev = weights[i]
            weights[i] = f.value*np.e/sum
            delta_z += abs(z_prev/weights[i])
