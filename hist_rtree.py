from collections import deque
from typing import List
from debug import draw_tree
import numpy as np
from rtree import index
import time
from bucket import *
from train_utils import *


property = index.Property()
global_buckets_dict = dict()
global_queries_rtree = index.Index()
identifier = 0
global_intersections = dict()


def deduplicate(buckets: List[Bucket]):
    buckets.sort(key=lambda x: x.mins+x.maxs)
    previous_bucket = buckets[0]
    unique_buckets = [previous_bucket]
    for current_bucket in buckets:
        if not are_coincide(previous_bucket, current_bucket):
            unique_buckets.append(current_bucket)
            previous_bucket = current_bucket
    return unique_buckets


def gen_global_rtree(buckets: List[Bucket]):
    for i, bucket in enumerate(buckets):
        global_queries_rtree.insert(i, bucket.coordinates)
    return global_queries_rtree


def get_intersection_for_each_query(querys: List[Bucket], rtree: index.Index):
    query_intersection = dict()
    for i, q in querys:
        intersection = set(rtree.intersection(q.coordinates))
        query_intersection[i] = intersection
    return query_intersection


def add_global_bucket(bucket: Bucket):
    # print("add_global_bucket")

    global global_buckets_dict, identifier

    global_buckets_dict[identifier] = bucket
    bucket.identifier = identifier
    bucket.composed.add(identifier)
    identifier += 1


def delete_global_bucket(bucket: Bucket):
    # print("delete_global_bucket")
    global global_buckets_dict, identifier
    del global_buckets_dict[bucket.identifier]


def gen_query_buckets(queries):
    buckets = [Bucket(q.mins, q.maxs, q.card)for q in queries]
    unique_buckets = deduplicate(buckets)
    unique_buckets.sort(key=lambda x: x.density)
    return unique_buckets


def find_small_parent(query: Bucket, root: Bucket):
    stack = deque([root])
    while stack:
        node = stack.pop()
        target = True
        for id in node.children:
            bucket = global_buckets_dict[id]
            if are_contain(bucket, query):
                stack.append(bucket)
                target = False
                break
        if target:
            return node


def ids_2_buckets(ids):
    return set([global_buckets_dict[id] for id in ids])


def sum_volume(buckets):
    return sum([bucket.volume for bucket in buckets])


def check_volume(hist):
    for cid in hist.children:
        bucket = global_buckets_dict[cid]
        check_volume(bucket)
        if not work_check_volume(bucket):
            print("error 3")


def work_check_volume(bucket):
    cbs = ids_2_buckets(bucket.composed)
    vol = sum_volume(cbs)
    return are_floats_equal(vol, bucket.cover_volume)


def cacl_valid_contains(overlap_contain_buckets: set, composed_ids: set):
    valid_contain_buckets = set()

    # 找到overlap_contain_buckets中合法的

    for bucket in overlap_contain_buckets:
        valid_flag = True
        for parent in bucket.parents:
            if parent.identifier in composed_ids:
                valid_flag = False
                break
        if valid_flag:
            valid_contain_buckets.add(bucket)
    return valid_contain_buckets


def merge_bucket_with_parent(bucket: Bucket):
    # print("merge")
    if bucket.identifier == 0:
        return
    delete_global_bucket(bucket)
    # 删除其孩子节点的父节点中有关bucket的信息
    for cid in bucket.children:
        child = global_buckets_dict[cid]
        child.parents.remove(bucket)

    # 将其孩子节点添加到bucket的父节点中
    for parent in bucket.parents:
        parent.merge_update(bucket, global_buckets_dict)


def feed_a_query_naive(query: Bucket, root: Bucket, overlaps: set, contains: set, disjoints: set):

    best_parent = True
    query_contain_child_buckets = set()
    query_contain_compose_ids = set()

    overlap_contain_buckets = set()
    overlap_contain_compose_ids = set()

    to_remove_compose_ids = set()

    for id in root.children.copy():
        bucket = global_buckets_dict[id]
        if id in overlaps:
            if are_coincide(query, bucket.overlap_with_query.data):  # query已经被包含过
                best_parent = False
                cret_dataset = bucket.overlap_with_query.dataset
                cret_composed = bucket.overlap_with_query.composed_set
                to_remove_compose_ids |= bucket.overlap_with_query.to_remove
            else:
                overlap_contain_compose_ids |= bucket.overlap_with_query.composed_set
                dataset = bucket.overlap_with_query.dataset
                if isinstance(dataset, set):
                    overlap_contain_buckets |= dataset
                else:
                    overlap_contain_buckets.add(dataset)
                to_remove_compose_ids |= bucket.overlap_with_query.to_remove
        elif id in contains:
            query_contain_child_buckets.add(bucket)
            query_contain_compose_ids |= bucket.composed
        elif id in disjoints:
            continue
        else:  # 未被查过
            if are_disjoint(query, bucket):
                disjoints.add(id)
            else:
                if are_contain(query, bucket):
                    query_contain_child_buckets.add(bucket)
                    query_contain_compose_ids |= bucket.composed
                    contains.add(id)
                else:
                    if are_contain(bucket, query):  # bucket包含query
                        best_parent = False
                        bucket.overlap_with_query = root.overlap_with_query
                        cret_dataset, cret_composed, cret_to_remove = feed_a_query_naive(
                            query, bucket, overlaps, contains, disjoints)
                        bucket.overlap_with_query.dataset = cret_dataset
                        bucket.overlap_with_query.composed_set = cret_composed
                        bucket.overlap_with_query.to_remove = cret_to_remove
                        to_remove_compose_ids |= cret_to_remove
                    else:  # bucket不包含query
                        new_overlap = get_overlap(query, bucket)
                        bucket.overlap_with_query = Container()
                        bucket.overlap_with_query.data = new_overlap
                        oret_dataset, oret_composed, oret_to_remove = feed_a_query_naive(
                            new_overlap, bucket, overlaps, contains, disjoints)
                        bucket.overlap_with_query.dataset = oret_dataset
                        bucket.overlap_with_query.composed_set = oret_composed
                        bucket.overlap_with_query.to_remove = oret_to_remove
                        overlap_contain_compose_ids |= oret_composed
                        if isinstance(oret_dataset, set):
                            overlap_contain_buckets |= oret_dataset
                        else:
                            overlap_contain_buckets.add(oret_dataset)
                        to_remove_compose_ids |= oret_to_remove
                    overlaps.add(id)
    if best_parent:
        query_compose_ids = overlap_contain_compose_ids | query_contain_compose_ids
        query_compose_buckets = ids_2_buckets(query_compose_ids)
        compose_volume = sum_volume(query_compose_buckets)
        valid_overlap_contain_buckets = cacl_valid_contains(overlap_contain_buckets, query_compose_ids)
        valid_contain_buckets = valid_overlap_contain_buckets | query_contain_child_buckets

        if are_floats_equal(compose_volume, query.volume):
            root.composed |= query_compose_ids
            root.composed -= to_remove_compose_ids
            return valid_contain_buckets, query_compose_ids, to_remove_compose_ids
        else:
            add_global_bucket(query)
            query.composed |= query_compose_ids
            root.composed |= query.composed
            root.composed -= to_remove_compose_ids
            # 删除root的children中属于query的
            root.delete_contains(query_contain_child_buckets)
            # 向query中添加bucket
            for item in valid_contain_buckets:
                query._init_add(item)
            query.volume -= compose_volume
            root.add_for_overlap(query)
            if is_close_to_zero(root.volume):
                merge_bucket_with_parent(root)
                to_remove_compose_ids.add(root.identifier)
            return query, query.composed.copy(), to_remove_compose_ids

    else:
        root.composed |= cret_composed
        root.composed -= to_remove_compose_ids
        return cret_dataset, cret_composed, to_remove_compose_ids


def get_contains_naive(parent, query):
    init_contains = set()
    for cid in parent.children:
        bucket = global_buckets_dict[cid]
        if are_contain(query, bucket):
            init_contains |= bucket.composed
    return init_contains


def feed_a_query_root(query: Bucket, root: Bucket):
    parent = find_small_parent(query, root)  # 包含query的最小的bucket
    init_contains = get_contains_naive(parent, query)
    init_overlaps = set()
    init_disjoint = set()
    feed_a_query_naive(query, parent, init_overlaps, init_contains, init_disjoint)
    #find_overlap_with_query(query, parent, overlaps,visited, query_contains)



class Max_Entropy(object):
    def __init__(self):
        pass
    
    







def construct_histogram(queries, mins, maxs, num_tuples):
    print("Start Generate the Histogram...")
    global global_buckets_dict, global_queries_rtree, identifier
    start = time.time()
    root_bucket = Bucket(mins, maxs, num_tuples)
    add_global_bucket(root_bucket)
    input_buckets = gen_query_buckets(queries)[:50]
    query_rtree = gen_global_rtree(input_buckets)

    num_buckets = len(input_buckets)
    for i, bucket in enumerate(input_buckets):
        print(f"Construct Histogram Step [{i+1}/{num_buckets}]")
        feed_a_query_root(bucket, root_bucket)
    end = time.time()
    print("Generate Hisogram Time Cost:{}".format(end-start))

    for key, bucket in global_buckets_dict.items():
        if bucket.volume <= 0:
            print("volume error", bucket.volume)
    return root_bucket


def test():
    """ queries = [Bucket(mins=[0.1, 0.1], maxs=[0.6, 0.6]),
               Bucket(mins=[0.3, 0.65], maxs=[0.5, 0.7]),
               Bucket(mins=[0.3, 0.5], maxs=[0.5, 0.8]),
               Bucket(mins=[0.1, 0.1], maxs=[0.6, 0.6]), ] """

    """ queries = [
        Bucket(mins=[0.1, 0.1], maxs=[0.6, 0.6], card=100),
        Bucket(mins=[0.3, 0.3], maxs=[0.8, 0.8], card=150),
        Bucket(mins=[0.7, 0.1], maxs=[0.8, 0.8], card=200),
        Bucket(mins=[0.3, 0.2], maxs=[0.6, 0.8], card=250),
        # Bucket(mins=[0.3, 0.2], maxs=[0.6, 0.8]),
        # Bucket(mins=[0.1, 0.1], maxs=[0.6, 0.6]),
        # Bucket(mins=[0.1, 0.65], maxs=[0.5, 0.7]),
    ] """

    queries = [
        Bucket(mins=[0.1, 0.5], maxs=[0.6, 0.9], card=6),
        Bucket(mins=[0.4, 0.2], maxs=[0.9, 0.8], card=10),
        Bucket(mins=[0.2, 0.3], maxs=[0.7, 0.7], card=20),
    ]

    hist = construct_histogram(queries, [0, 0], [1, 1], 1000)

    print(global_buckets_dict.keys())

    """ for i, c in enumerate(hist.children):
        # print("num", i)
        draw_tree(c, i) """
    draw_tree(hist, 100, global_buckets_dict)
    return hist


# test()
