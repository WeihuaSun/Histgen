from collections import OrderedDict
from typing import List, Dict
from debug import draw_tree
import numpy as np
from rtree import index
import time


from bucket import *
from utils import *

global_buckets_rtree = index.Index()

global_queries_rtree = index.Index()

global_buckets_dict = dict()

identifier = 0

p = index.Property()


def find_parent(query, root) -> Bucket:
    for child in root.children.values():
        if are_contain(child, query):
            return find_parent(query, child)
        elif not are_disjoint(child, query):
            return root
    return root


def deduplicate(buckets):
    buckets.sort(key=lambda x: x.mins+x.maxs)
    previous_bucket = buckets[0]
    unique_buckets = [previous_bucket]
    for current_bucket in buckets:
        if not are_coincide(previous_bucket, current_bucket):
            unique_buckets.append(current_bucket)
            previous_bucket = current_bucket
    return unique_buckets


def intersect_with_children(root: Bucket, query: Bucket):
    # 输入一个query_bucket,输出和其相交的所有的bucket
    coordinates = query.mins+query.maxs
    contains = set(root.rtree.contains(coordinates))  # query包含的bucket
    intersections = set(root.rtree.intersection(coordinates))
    overlaps = intersections-contains
    return overlaps, contains


def feed_a_overlap(query: Bucket, root: Bucket):
    parent = find_parent(query, root)
    if are_coincide(parent, query):
        return parent
    overlaps, contains = intersect_with_children(parent, query)
    # parent中没有和这个查询相交或者包含的，则将这个bucket作为parent的孩子节点
    if len(overlaps) == 0 and len(contains) == 0:
        parent.add_for_overlap(query)
        if is_close_to_zero(parent.volume):
            merge_bucket_with_parent(parent)
        return query

    # 处理contains,即query包含的一些bucket,如有必要，需要将这些bucket作为query的孩子，并且从parent中移除这些bucket
    # overlap contain的bucket不会被query分割，应该也不会被处理（？）为了避免可能的重复操作，还是更新feature_id
    cur_contains = []
    for cid in contains:
        bucket = parent.children[cid]
        cur_contains.append(bucket)
    global global_buckets_rtree, global_buckets_dict
    query.add_for_query(cur_contains, global_buckets_rtree, global_buckets_dict)
    if is_close_to_zero(query.volume):  # 如果这些包含的bucket恰好可以组成query,则不添加query
        return

    # S4:处理和query相交的bucket
    tmp_rtree = index.Index(properties=p)
    tmp_overlap_buckets = []
    for oid in overlaps:
        bucket = parent.children[oid]
        if are_disjoint(bucket, query):  # rtree实现的overlap似乎和我们的overlap有歧义，仅仅边相交的，我们不当作overlap
            continue
        overlap = get_overlap(query, bucket)  # 相交区域
        tmp_overlap_buckets.append((oid, (overlap, bucket)))
        tmp_rtree.insert(oid, overlap.coordinates)

    overlaps_sorted = OrderedDict(sorted(tmp_overlap_buckets, key=lambda x: x[1][0].volume))
    valid_overlaps = []
    while overlaps_sorted:
        oid, (overlap, bucket) = overlaps_sorted.popitem()
        valid_overlaps.append((overlap, bucket))
        this_contains = list(tmp_rtree.contains(overlap.coordinates))
        for id in this_contains:
            (doverlap, dbucket) = overlaps_sorted.pop(id)
            tmp_rtree.delete(id, doverlap.coordinates)
    composed = []
    for overlap, bucket in valid_overlaps:
        ret = feed_a_overlap(overlap, bucket)
        if isinstance(ret, list):
            composed += ret
        else:
            composed.append(ret)
    global global_buckets_rtree, global_buckets_dict
    query.add_for_query(composed, global_buckets_rtree, global_buckets_dict)
    if is_close_to_zero(query.volume):
        return cur_contains+composed

    parent.delete_contains(contains, cur_contains)
    parent.add_for_overlap(query)
    if is_close_to_zero(parent.volume):
        merge_bucket_with_parent(parent)
    return query


def add_global_buckets():
    global global_buckets_dict, global_buckets_rtree, identifier


def feed_a_query(query: Bucket, root: Bucket):
    parent = find_parent(query, root)  # 找到某个Bucket,其包含query,但其字节的不包含query
    # S1:query和当前直方图中某个Bucket重合
    if are_coincide(parent, query):
        parent.cover_card = query.card
        query = parent  # 定位这个bucket
        return

    overlaps, contains = intersect_with_children(
        parent, query)  # 从parent中找到其直接overlap和contain的Bucket的rid

    # S2:parent中没有和这个查询相交或者包含的，则将这个bucket作为parent的孩子节点
    if len(overlaps) == 0 and len(contains) == 0:
        parent.add_for_overlap(query)
        if is_close_to_zero(parent.volume):
            merge_bucket_with_parent(parent)
        return

    # S3:处理被query包含的bucket，将这些bucket作为query的孩子，将其从parent中去除
    cur_contains = []
    for cid in contains:
        bucket = parent.children[cid]
        cur_contains.append(bucket)
    global global_buckets_rtree, global_buckets_dict
    query.add_for_query(cur_contains, global_buckets_rtree, global_buckets_dict)
    if is_close_to_zero(query.volume):  # 如果这些包含的bucket恰好可以组成query,则不添加query
        return

    # S4:处理和query相交的bucket
    tmp_rtree = index.Index(properties=p)
    tmp_overlap_buckets = []
    for oid in overlaps:
        bucket = parent.children[oid]
        if are_disjoint(bucket, query):  # rtree实现的overlap似乎和我们的overlap有歧义，仅仅边相交的，我们不当作overlap
            continue
        overlap = get_overlap(query, bucket)  # 相交区域
        tmp_overlap_buckets.append((oid, (overlap, bucket)))
        tmp_rtree.insert(oid, overlap.coordinates)

    overlaps_sorted = OrderedDict(sorted(tmp_overlap_buckets, key=lambda x: x[1][0].volume))
    valid_overlaps = []
    while overlaps_sorted:
        oid, (overlap, bucket) = overlaps_sorted.popitem()
        valid_overlaps.append((overlap, bucket))
        this_contains = list(tmp_rtree.contains(overlap.coordinates))
        for id in this_contains:
            (doverlap, dbucket) = overlaps_sorted.pop(id)
            tmp_rtree.delete(id, doverlap.coordinates)
    composed = []
    for overlap, bucket in valid_overlaps:
        ret = feed_a_overlap(overlap, bucket)
        if isinstance(ret, list):
            composed += ret
        else:
            composed.append(ret)
    global global_buckets_rtree, global_buckets_dict
    query.add_for_query(composed, global_buckets_rtree, global_buckets_dict)
    if is_close_to_zero(query.volume):
        return

    # 最终overlaps和contains都没有填充query，则将query添加为parent的孩子，并将parent中被query包含的孩子转换为query的孩子
    parent.delete_contains(contains, cur_contains)
    parent.add_for_overlap(query)
    if is_close_to_zero(parent.volume):
        merge_bucket_with_parent(parent)
    return


def merge_bucket_with_parent(bucket: Bucket):
    """当该某个Bucket由他的孩子Bucket全部填充,则删除这个Bucket,并将其孩子节点添加到其父节点中
    Args:
        bucket (Bucket):要被merge的节点
    """
    if bucket.parents:  # 如果没有父节点则跳过（root）
        # 删除其孩子节点的父节点中有关bucket的信息
        for child in bucket.children:
            del child.parents[bucket]

        # 将其孩子节点添加到bucket的父节点中
        for parent, bid in bucket.parents.items():
            parent.merge_update(bid, bucket)


def iter():
    pass


def freq():
    pass


def delete_zero():
    pass


def gen_global_rtree(buckets: List[Bucket]):
    global global_queries_rtree
    for i, bucket in enumerate(buckets):
        coordinates = bucket.mins+bucket.maxs
        global_queries_rtree.insert(i, coordinates)


def construct_histogram(queries, mins, maxs, num_tuples):
    p.dimension = len(queries[0].mins)
    global global_buckets_rtree, global_queries_rtree
    global_buckets_rtree.properties = p
    global_queries_rtree.properties = p
    root_bucket = Bucket(mins, maxs, num_tuples)
    print("Start Generate the Histogram ")
    start = time.time()
    input_buckets = [Bucket(q.mins, q.maxs, q.card)for q in queries]
    input_buckets = deduplicate(input_buckets)  # 删除重复的Bucket
    input_buckets.sort(key=lambda x: x.density)  # 按照密度对Bucket进行排序，密度大的在最后
    num_buckets = len(input_buckets)
    gen_global_rtree(input_buckets)
    # gis_solver = Generalized_Iterative_Scaling()
    for i, bucket in enumerate(input_buckets):
        if i % 10 == 0:
            print(f"Construct Histogram Step [{i+1}/{num_buckets}]")
        feed_a_query(bucket, root_bucket)
        # iter()
        # freq()
        # delete_zero()
    end = time.time()
    print("Generate Hisogram Time Cost:{}".format(end-start))
    return root_bucket


def test():
    queries = [Bucket(mins=[0.1, 0.1], maxs=[0.6, 0.6]),
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
    ]
    hist = construct_histogram(queries, [0, 0], [1, 1], 1000)

    for i, c in enumerate(hist.children.values()):
        print("num", i)
        draw_tree(c, i)
    draw_tree(hist, 100)
    return hist


test()
