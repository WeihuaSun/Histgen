from collections import OrderedDict, deque
from typing import List, Dict
from debug import draw_tree
import numpy as np
from rtree import index
import time


from bucket import *
from train_utils import *


p = index.Property()
p.dimension = 14
global_buckets_rtree = index.Index(properties=p)

global_queries_rtree = index.Index(properties=p)

global_buckets_dict = dict()

identifier = 1

feature_id = 1


def deduplicate(buckets):
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


def add_global_bucket(bucket: Bucket):
    # print("add_global_bucket")
    global global_buckets_dict, global_buckets_rtree, identifier
    global_buckets_rtree.insert(identifier, bucket.coordinates)
    global_buckets_dict[identifier] = bucket
    bucket.identifier = identifier
    identifier += 1


def delete_global_bucket(bucket: Bucket):
    # print("delete_global_bucket")
    global global_buckets_dict, global_buckets_rtree, identifier
    global_buckets_rtree.delete(bucket.identifier, bucket.coordinates)
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


def find_overlap_with_query(query: Bucket, root: Bucket, overlaps: set, visited: set):
    target = True  # 标记，指示root的孩子节点中是否有包含query的，若target=True，则root就是query的最小父节点，将query添加到root的孩子节点中
    overlap_contains_buckets = []
    for id in root.children:
        if id in overlaps:  # 和query相交
            bucket = global_buckets_dict[id]
            if id in visited:  # 已经被查过
                if are_coincide(query, bucket.overlap_with_query.data):  # query已经被包含过
                    root.overlap_with_query = bucket.overlap_with_query
                    target = False
                    ret = root.overlap_with_query.dataset
                    break
                else:
                    overlap_contains_buckets.append(
                        bucket.overlap_with_query.dataset)
            else:  # 未被查过
                if are_contain(bucket, query):  # bucket包含query
                    target = False
                    ret = find_overlap_with_query(
                        query, bucket, overlaps, visited)
                    bucket.overlap_with_query = root.overlap_with_query

                else:  # bucket不包含query
                    new_overlap = get_overlap(query, bucket)
                    ret = find_overlap_with_query(
                        new_overlap, bucket, overlaps, visited)
                    overlap_contains_buckets.append(ret)
                    bucket.overlap_with_query.data = new_overlap
                bucket.overlap_with_query.dataset = ret
                visited.add(id)
    if target:  # 其孩子节点均不包含input
        query_contains, contains_volume = check_cover(query)
        this_contain = query_contains & root.children
        this_contain_buckets = set([global_buckets_dict[id]
                                   for id in this_contain])
        valid_contains = cacl_valid_contains(
            this_contain_buckets, overlap_contains_buckets, query_contains)

        if are_floats_equal(contains_volume, query.volume):
            return valid_contains
        else:  # 添加
            add_global_bucket(query)
            # 删除root的children中属于query的
            root.delete_contains(this_contain_buckets)
            # 向query中添加bucket
            for item in valid_contains:
                query._init_add(item)
            query.volume -= contains_volume
            root.add_for_overlap(query)
            if is_close_to_zero(root.volume):
                merge_bucket_with_parent(root)
            return query
    else:
        return ret


def cacl_valid_contains(this_contains: set, overlap_contains_buckets: set, query_contains: set):
    overlap_contains_set = set()
    for item in overlap_contains_buckets:
        if isinstance(item, set):
            overlap_contains_set |= item
        else:
            overlap_contains_set.add(item)
    valid_overlaps = set()
    for bucket in overlap_contains_set:
        valid_flag = True
        for parent in bucket.parents:
            if parent.identifier in query_contains:
                valid_flag = False
                break
        if valid_flag:
            valid_overlaps.add(bucket)
    valid_overlaps |= this_contains
    return valid_overlaps & query_contains


def merge_bucket_with_parent(bucket: Bucket):
    """当该某个Bucket由他的孩子Bucket全部填充,则删除这个Bucket,并将其孩子节点添加到其父节点中
    Args:
        bucket (Bucket):要被merge的节点
    """
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


def check_cover(query: Bucket):
    query_contains = set(global_buckets_rtree.contains(query.coordinates))
    contains_volume = np.sum(
        [global_buckets_dict[id].volume for id in query_contains])
    return query_contains, contains_volume


def feed_a_query_root(query: Bucket, root: Bucket):
    global global_buckets_rtree, global_buckets_dict

    # 检查当前query是否已经被覆盖
    query_contains, contains_volume = check_cover(query)

    # 没有被覆盖

    if not are_floats_equal(contains_volume, query.volume):
        parent = find_small_parent(query, root)  # 包含query的最小的bucket
        assert parent is not None

        parent_contains = set(global_buckets_rtree.contains(
            parent.coordinates))
        intersections = set(
            global_buckets_rtree.intersection(query.coordinates))

        overlap_to_remove = set()

        overlaps = parent_contains & intersections - query_contains  # 和query相交的所有bucket

        for id in overlaps:
            bucket = global_buckets_dict[id]
            if are_disjoint(bucket, query):
                overlap_to_remove.add(id)
        overlaps -= overlap_to_remove

        find_overlap_with_query(query, parent, overlaps, visited=set())


def construct_histogram(queries, mins, maxs, num_tuples):
    print("Start Generate the Histogram...")
    global feature_id, global_buckets_dict
    p.dimension = len(queries[0].mins)
    root_bucket = Bucket(mins, maxs, num_tuples)
    global_buckets_dict[0] = root_bucket
    start = time.time()
    input_buckets = gen_query_buckets(queries)
    gen_global_rtree(input_buckets)
    num_buckets = len(input_buckets)
    for i, bucket in enumerate(input_buckets):
        feature_id = i+1
        print(f"Construct Histogram Step [{i+1}/{num_buckets}]")
        feed_a_query_root(bucket, root_bucket)
    end = time.time()
    print("Generate Hisogram Time Cost:{}".format(end-start))
    return root_bucket


def test():
    """ queries = [Bucket(mins=[0.1, 0.1], maxs=[0.6, 0.6]),
               Bucket(mins=[0.3, 0.65], maxs=[0.5, 0.7]),
               Bucket(mins=[0.3, 0.5], maxs=[0.5, 0.8]),
               Bucket(mins=[0.1, 0.1], maxs=[0.6, 0.6]), ]

    queries = [
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
