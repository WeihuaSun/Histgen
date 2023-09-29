from rtree import index
import numpy as np


class Bucket:
    def __init__(self, mins, maxs, card=0):
        self.mins = mins
        self.maxs = maxs
        self.coordinates = mins+maxs
        self.card = card
        self.cover_card = card
        self.children = dict()
        self.parents = dict()
        self.volume = cacl_volume(mins, maxs)
        self.cover_volume = self.volume
        self.density = self.card/self.volume
        p = index.Property()
        p.dimension = len(self.mins)
        self.rtree = index.Index(properties=p)
        self.contain_buckets = set()
        self.crid = 0
        self.constraints = []  # 该Bucket由哪些约束组成

    def add_for_query(self, input, global_buckets_rtree, global_buckets_dict):
        # assert isinstance(input, list)
        for b in input:
            self._init_add(b)
        # update volume
        contains = set(global_buckets_rtree.contains(self.coordinates))
        # update method 1
        self.volume -= np.sum([global_buckets_dict[bid].volume for bid in contains])
        # self.composed_ids = contains

    def add_for_contain(self, input, global_buckets_rtree, global_buckets_dict):
        for b in input:
            self._init_add(b)
        contains = set(global_buckets_rtree.contains(self.coordinates))
        # update method 1
        self.volume -= np.sum([global_buckets_dict[bid].volume for bid in contains])

    def add_for_overlap(self, input):
        def add(bucket):
            self._init_add(bucket)
        if isinstance(input, list):
            # overlap is composed of some children buckets of dest bucket
            # volume does not change
            pass
        else:
            add(input)
            self.volume -= input.volume

    def _init_add(self, bucket):
        coordinates = bucket.mins+bucket.maxs
        bucket.parents[self] = self.crid
        self.rtree.insert(self.crid, coordinates)
        self.children[self.crid] = bucket
        self.crid += 1

    def delete_a_child(self, bucket, bid=None):
        if bid == None:
            bid = list(self.children.keys())[
                list(self.children.values()).index(bucket)]
        coordinates = bucket.mins+bucket.maxs
        self.rtree.delete(bid, coordinates)
        del self.children[bid]

    def delete_contains(self, bids, buckets):
        for bid, bucket in zip(bids, buckets):
            self.delete_a_child(bucket, bid)

    def update_constraints(self, new_constraints):
        self.constraints.append(new_constraints)

    def merge_update(self, bucket, bid):
        children = list(bucket.children.values())
        self.delete_a_child(bucket, bid)
        for c in children:
            coordinates = c.mins+c.maxs
            c.parents[self] = self.crid
            self.rtree.insert(self.crid, coordinates)
            self.children[self.crid] = bucket
            self.composed_set |= bucket.composed_set
            self.crid += 1


def cacl_volume(mins, maxs):
    # 计算超立方体(Bucket)的体积
    vol = 1.
    for min, max in zip(mins, maxs):
        vol *= (max-min)
    return vol


def are_coincide(a, b):
    # a和b是否重合
    if a.mins == b.mins and a.maxs == b.maxs:
        return True
    return False


def get_overlap(a, b):
    # 两个超立方体重叠部分
    mins = [max(a_m, b_m) for a_m, b_m in zip(a.mins, b.mins)]
    maxs = [min(a_m, b_m) for a_m, b_m in zip(a.maxs, b.maxs)]
    overlap = Bucket(mins, maxs, 1)
    return overlap


def are_contain(a, b):
    # a是否包含b
    for a_min, b_min in zip(a.mins, b.mins):
        if a_min > b_min:
            return False
    for a_max, b_max in zip(a.maxs, b.maxs):
        if a_max < b_max:
            return False
    return True


def are_disjoint(a, b):
    # 两个立方体是否分离
    for a_min, b_max in zip(a.mins, b.maxs):
        if a_min >= b_max:
            return True
    for a_max, b_min in zip(a.maxs, b.mins):
        if a_max <= b_min:
            return True
    return False
