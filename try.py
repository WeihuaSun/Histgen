""" from rtree import index

rtree = index.Index()


rtree.insert(0,[0,0,1,1])
rtree.insert(1,[1,0,2,1])

print(set(rtree.intersection([0,0,1,1]))) """

class Container:
    def __init__(self):
        self.dataset = set()
        self.data = None


class Bucket:
    def __init__(self, id=0):
        self.identifier = id
        self.overlap_with_query = Container()

def test():
    a = Bucket(1)
    b = Bucket(2)
    c = Bucket(3)
    set_a = {a,b}
    
    set_b = set_a.copy()
    
    set_a.add(c)
    
    print(set_a)
    print(set_b)

test()



