from rtree import index
import numpy as np


class Bucket:
    def __init__(self, mins, maxs, card=0):
        self.identifier = 0
        self.mins = mins
        self.maxs = maxs
        self.coordinates = mins+maxs
        self.card = card
        # self.cover_card = card
        self.children = set()
        self.hist_cs = set()
        self.parents = set()
        self.volume = cacl_volume(mins, maxs)
        self.cover_volume = self.volume
        self.density = self.card/self.volume
        p = index.Property()
        p.dimension = len(self.mins)
        self.rtree = index.Index(properties=p)
        self.check_id = 0
        self.feature_cover = 0
        self.feature_father = 0
        self.featured_children = set()
        # self.contain_buckets = set()
        # self.constraints = []  # 该Bucket由哪些约束组成

    def add_for_query(self, input, global_buckets_rtree, global_buckets_dict, input_volume=0):
        if isinstance(input, list):
            for b in input:
                self._init_add(b)
        else:
            self._init_add(input)
        # update volume
        contains = set(global_buckets_rtree.contains(self.coordinates))
        self.volume = self.cover_volume - \
            np.sum([global_buckets_dict[bid].volume for bid in contains]
                   )  # update method 1

    def add_for_contain(self, input, global_buckets_rtree, global_buckets_dict):
        for b in input:
            self._init_add(b)
        contains = set(global_buckets_rtree.contains(self.coordinates))
        # update method 1
        self.volume -= np.sum([global_buckets_dict[bid].volume for bid in contains])

    def add_for_overlap(self, input):
        # print("add_for_overlap")
        self._init_add(input)
        self.volume -= input.volume

    def _init_add(self, bucket):
        assert bucket.identifier != 0
        identifier = bucket.identifier
        bucket.parents.add(self)
        if identifier not in self.hist_cs:
            self.rtree.insert(identifier, bucket.coordinates)
        self.children.add(identifier)
        self.hist_cs.add(identifier)

    def delete_a_child(self, bucket):
        assert bucket.identifier != 0
        identifier = bucket.identifier
        bucket.parents.remove(self)
        self.children.remove(identifier)
        # self.rtree.delete(identifier, bucket.coordinates)

    def delete_contains(self, buckets):
        for bucket in buckets:
            self.delete_a_child(bucket)

    """ def update_constraints(self, new_constraints):
        self.constraints.append(new_constraints) """

    def merge_update(self, bucket, buckets_dict):
        # print("merge_update")
        identifier = bucket.identifier
        assert identifier != 0
        self.children.remove(identifier)
        # self.rtree.delete(identifier, bucket.coordinates)
        for c in bucket.children:
            child = buckets_dict[c]
            self._init_add(child)


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
