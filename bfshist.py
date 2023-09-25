# Bucket overlapped multidimensional histograms
import numpy as np
from rtree import index
import time
import pulp
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Bucket(object):
    def __init__(self, mins, maxs, card=0, level=0, origin=False):
        self.mins = mins
        self.maxs = maxs
        self.card = card
        self.children = []
        #self.overlap_children = []
        self.volume = cacl_volume(mins, maxs)
        self.rid = 0
        #self.composed_ids = {}
        self.father = set()
        global Uid
        Uid += 1
        self.uid = Uid
        self.level = level
        self.origin = origin

    def add_child(self, child):
        self.children.append(child)

    def search(self, composed_ids):
        pass

    """ def __eq__(self, __value: object) -> bool:
        return self.uid == __value.uid

    def __hash__(self) -> int:
        return self.uid """


def cacl_volume(mins, maxs):
    # 计算超立方体(Bucket)的体积
    vol = 1.
    for min, max in zip(mins, maxs):
        vol *= (max-min)
    return vol


def float_equal(a, b):
    return abs(a-b) < 1e-6


def are_coincide(a, b):
    # a和b是否重合
    if a.mins == b.mins and a.maxs == b.maxs:
        return True
    return False


def get_overlap(a, b):
    # 两个超立方体重叠部分
    mins = [max(a_m, b_m) for a_m, b_m in zip(a.mins, b.mins)]
    maxs = [min(a_m, b_m) for a_m, b_m in zip(a.maxs, b.maxs)]
    overlap = Bucket(mins, maxs, 0)
    #overlap.composed_ids = a.composed_ids+b.composed_ids
    return overlap


Level = 0
Uid = 0
p = index.Property()


def delete_same(sorted_bucket_list):
    distinct_buckets = []
    while sorted_bucket_list:
        big = sorted_bucket_list.pop()
        big_vol = big.volume
        same_volumes = [big]
        while sorted_bucket_list:
            cur = sorted_bucket_list[-1]
            cur_vol = cur.volume
            if cur_vol == big_vol:
                to_add = True
                sorted_bucket_list.pop()
                for i, sv in enumerate(same_volumes):
                    if are_coincide(cur, sv):
                        to_add = False
                        if cur.origin:
                            # Replace not origin bucket
                            same_volumes.pop(i)
                            same_volumes.append(cur)
                            cur.father = cur.father | sv.father
                        else:
                            sv.father = cur.father | sv.father
                        break
                if to_add:
                    same_volumes.append(cur)
            else:
                break
        distinct_buckets += same_volumes
    distinct_buckets.reverse()
    return distinct_buckets


def choose_this_level(sorted_buckets: list, rtree: index.Index):
    # Note:从Rtree中删除Node的耗费的时间太久
    print("Choose Level...")
    low_level_buckets = []  # Be contained
    this_level_buckets = []  # Not be contained
    copy_sorted_buckets = sorted_buckets[:]
    have_checked = set()  # have checked ids
    start = time.time()
    num_buckets = len(sorted_buckets)
    sorted_buckets.reverse()
    for i, big_bucket in enumerate(sorted_buckets):
        rid = num_buckets-i-1
        if i % 1000 == 0:
            print("Step[{}/{}]".format(i, num_buckets))
        if rid not in have_checked:
            have_checked.add(rid)
            this_level_buckets.append(big_bucket)
            coordinates = big_bucket.mins+big_bucket.maxs
            #rtree.delete(big_bucket.rid, coordinates)
            contains = list(rtree.contains(coordinates))
            for b in contains:
                bk = copy_sorted_buckets[b]
                if bk.rid != rid:
                    # if b not in have_checked:
                    rtree.delete(bk.rid, bk.mins+bk.maxs)
                    # sorted_buckets.remove(bk)
                    have_checked.add(b)
                    low_level_buckets.append(bk)
                    bk.father = set([big_bucket])
                    # else:
                    bk.father.add(big_bucket)
    end = time.time()
    print("Time Cost:{}".format(end-start))
    print("Number of This Level Bucket:{} ".format(len(this_level_buckets)))
    print("Number of Lower Level Buckets:{}".format(len(low_level_buckets)))
    return this_level_buckets, low_level_buckets


def get_overlap_each_other(buckets: list, rtree: index.Index):
    print("Get Overlaps of This Level Buckets...")
    num_buckets = len(buckets)
    overlap_buckets = []
    start = time.time()
    for i in range(num_buckets-1):
        if i % 100 == 0:
            print("Step[{}/{}]".format(i, num_buckets))
        bi = buckets[i]
        coordinates = bi.mins+bi.maxs
        intersections = set(list(rtree.intersection(coordinates)))
        for j in range(i+1, num_buckets):
            bj = buckets[j]
            if bj.rid in intersections:
                overlap_ij = get_overlap(bi, bj)
                overlap_ij.father.add(bi)
                overlap_ij.father.add(bj)
                overlap_buckets.append(overlap_ij)
    end = time.time()
    print("Time Cost:{}".format(end-start))
    print("Overlap Buckets:{}".format(len(overlap_buckets)))
    return overlap_buckets


def construct_rtree(buckets):
    print("Generate Rtree Index...")
    start = time.time()
    buckets_dict = dict()
    rtree = index.Index(properties=p)
    for i, query_bucket in enumerate(buckets):
        query_bucket.rid = i
        buckets_dict[i] = query_bucket
        coordinates = query_bucket.mins+query_bucket.maxs
        rtree.insert(i, coordinates)
    end = time.time()
    print("Done! Time Cost:{}".format(end-start))
    return rtree, buckets_dict


def gen_hist_child(input_buckets: list, low_level_buckets: list):
    all_level = input_buckets+low_level_buckets
    global Level
    Level += 1
    print("Level {} Histogram Generation...".format(Level))
    all_level.sort(key=lambda x: x.volume, reverse=False)
    print("Before Delete Same Buckets:{}".format(len(all_level)))
    all_level = delete_same(all_level)  # Step1:删除相同的Bucket,将他们的possible_father 组合在一起
    num_bucket = len(all_level)
    print("Number of Bucket to Process(After Delete Same): {}".format(num_bucket))

    # Step 2:构建Rtree,相同Level或者出现的更低Level的Buckets
    rtree, bucket_dict = construct_rtree(all_level)
    # Step3:分层次，选择出是当前层次的还是更低层次的
    this_level_buckets, new_low_level_buckets = choose_this_level(all_level, rtree)

    for b in all_level:
        if b in this_level_buckets:
            for f in b.father:
                f.add_child(b)
    next_level_buckets = get_overlap_each_other(this_level_buckets, rtree)  # Step4:当前层次的Bucket计算Overlap
    """
    Step 3 和Step 4能合并么?
    """
    print("Number of Next Level Buckets:{}".format(len(next_level_buckets)))
    del rtree
    if next_level_buckets or low_level_buckets:
        gen_hist_child(next_level_buckets, new_low_level_buckets)


def gen_hist_root(queries, mins, maxs, card):
    """
    Generate Histogram BFS
    """
    root_bucket = Bucket(mins, maxs, card)
    num_queries = len(queries)
    p.dimension = len(queries[0].mins)
    print("Generate Histogram...")
    print("Number of Queries:{}".format(num_queries))
    start = time.time()
    input_buckets = [Bucket(q.mins, q.maxs, q.card, level=0, origin=True) for q in queries]
    input_buckets.sort(key=lambda x: x.volume, reverse=False)  # Buckets with biggest volume loc at the end of the List
    for i, b in enumerate(input_buckets):
        b.father.add(root_bucket)
    input_buckets = delete_same(input_buckets)  # Remove Buckets with same bounds #census:18638
    input_buckets.pop()
    gen_hist_child([], input_buckets)
    end = time.time()
    print("Generate Hisogram Time Cost:{}".format(end-start))
    return root_bucket
