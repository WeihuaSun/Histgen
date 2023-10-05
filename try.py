from rtree import index

rtree = index.Index()

rtree.insert(0,[0,0,1,1])
rtree.insert(1,[1,0,2,1])

print(set(rtree.intersection([0,0,1,1])))
