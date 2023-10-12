import numpy as np

class Bucket:
    def __init__(self,volume,compose):
        self.compose = compose
        self.volume = volume
class Featrue:
    def __init__(self, value) -> None:
        self.compose = []
        self.value = value


def cacl_bucket(volume, weights, compose):
    compose_weight = np.array([weights[i] for i in compose])
    density = np.prod(compose_weight)
    num = density*volume
    return num


# Generalized Iterative Scaling
def gis(weights, features, eps=1e-9):
    delta_z = eps
    delta_z_prev = 10*eps
    while abs(delta_z_prev-delta_z) > eps:
        delta_z_prev = delta_z
        delta_z = 0
        for i, f in enumerate(features):
            sum = np.sum(np.array([cacl_bucket(b.volume, weights,
                         b.compose) for b in f.compose]))
            sum /= weights[i]
            z_prev = weights[i]
            weights[i] = f.value*np.e/sum
            delta_z += abs(z_prev/weights[i])
    



def test0():
    bucket0 = Bucket(24,[0])
    bucket1 = Bucket(4,[0,1])
    bucket2 = Bucket(4,[0,1,2])
    bucket3 = Bucket(8,[0,2])
    
    feature1 = Featrue(50)
    feature1.compose = [bucket0,bucket1,bucket2,bucket3]

    feature2 = Featrue(8)#density = 1
    feature2.compose = [bucket1,bucket2]

    feature3 = Featrue(15)#density = 2
    feature3.compose = [bucket2,bucket3]
    
    features = [feature1,feature2,feature3]

    weights = [1 for _ in features]

    gis(weights,features)
    buckets_list = [bucket0,bucket1,bucket2,bucket3]
    for bucket in buckets_list:
        print(cacl_bucket(bucket.volume,weights,bucket.compose)/np.e)





def test1():
    bucket0 = Bucket(20,[0]) 
    bucket1 = Bucket(4,[0,1])
    bucket2 = Bucket(4,[0,1,2])
    bucket3 = Bucket(7,[0,2])
    bucket4 = Bucket(1,[0,2,3])
    bucket5 = Bucket(4,[0,3])


    feature1 = Featrue(50)
    feature1.compose = [bucket0,bucket1,bucket2,bucket3,bucket4,bucket5]


    feature2 = Featrue(8)#density = 1
    feature2.compose = [bucket1,bucket2]

    feature3 = Featrue(24)#density = 2
    feature3.compose = [bucket2,bucket3,bucket4]

    feature4 = Featrue(24)
    feature4.compose = [bucket4,bucket5]

    features = [feature1,feature2,feature3,feature4]

    weights = [1 for _ in features]

    gis(weights,features)

    print(weights)

    buckets_list = [bucket0,bucket1,bucket2,bucket3,bucket4,bucket5]

    for bucket in buckets_list:
        print(cacl_bucket(bucket.volume,weights,bucket.compose)/np.e)


test0()

