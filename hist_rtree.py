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


@profile
def find_overlap_with_query(query: Bucket, root: Bucket, overlaps: set, visited: set, query_contains: set):
    target = True  # 标记，指示root的孩子节点中是否有包含query的，若target=True，则root就是query的最小父节点，将query添加到root的孩子节点中
    old_overlap_contains_buckets = []  # 保存root中的child和query相交产生的孩子
    new_overlap_contains_buckets = []   # 保存root中的child和query相交产生的孩子
    this_visited = set()
    for id in root.children.copy():
        if id in overlaps:  # 和query相交
            bucket = global_buckets_dict[id]
            if id in visited:  # 已经被查过
                if are_coincide(query, bucket.overlap_with_query.data):  # query已经被包含过
                    target = False
                    cret = bucket.overlap_with_query.dataset
                    ccover = 0
                    cset = {}
                    # root.overlap_with_query.data = bucket.overlap_with_query.data
                    # root.overlap_with_query.dataset = bucket.overlap_with_query.dataset
                else:
                    old_overlap_contains_buckets.append(
                        bucket.overlap_with_query)
            else:  # 未被查过
                if are_contain(bucket, query):  # bucket包含query
                    target = False
                    bucket.overlap_with_query = root.overlap_with_query
                    cret, ccover, cset = find_overlap_with_query(
                        query, bucket, overlaps, visited, query_contains)
                    bucket.overlap_with_query.dataset = cret
                    bucket.overlap_with_query.delta_cover = ccover
                    bucket.overlap_with_query.delta_composed = cset
                else:  # bucket不包含query
                    new_overlap = get_overlap(query, bucket)
                    bucket.overlap_with_query = Container()
                    bucket.overlap_with_query.data = new_overlap
                    oret, ocover, oset = find_overlap_with_query(
                        new_overlap, bucket, overlaps, visited, query_contains)
                    bucket.overlap_with_query.dataset = oret
                    bucket.overlap_with_query.delta_cover = ocover
                    bucket.overlap_with_query.delta_composed = oset
                    new_overlap_contains_buckets.append(
                        bucket.overlap_with_query)
                visited.add(id)
    if target:  # 其孩子节点均不包含input

        overlap_contain_buckets = set()
        new_composed_ids = set()
        new_delta_composed = set()

        # 处理new_overlap_contains
        n_overlap_cover_volume = 0
        for item in new_overlap_contains_buckets:
            n_overlap_cover_volume += item.delta_cover
            dataset = item.dataset
            if isinstance(dataset, set):
                overlap_contain_buckets |= dataset
            else:
                overlap_contain_buckets.add(dataset)
            new_composed_ids |= item.delta_composed

        # 处理old_overlap_contains
        o_overlap_cover_volume = 0
        for item in old_overlap_contains_buckets:
            o_overlap_cover_volume += item.delta_cover
            dataset = item.dataset
            if isinstance(dataset, set):
                overlap_contain_buckets |= dataset
            else:
                overlap_contain_buckets.add(dataset)
            new_composed_ids |= item.delta_composed

        # 处理contains
        this_contains = query_contains & root.children   # child中被包含在query中的bucket的id
        this_contain_buckets = set([global_buckets_dict[id]
                                   for id in this_contains])  # buckets
        this_query_contains = set()
        for bucket in this_contain_buckets:
            this_query_contains |= bucket.composed

        remain_contains = this_query_contains-overlap_contain_buckets

        contains_volume = np.sum([global_buckets_dict[i] for i in remain_contains]
                                 ) + o_overlap_cover_volume+n_overlap_cover_volume

        new_delta_composed = new_composed_ids.copy()

        contains_volume = o_overlap_cover_volume+n_overlap_cover_volume

        if are_floats_equal(contains_volume, query.volume):
            return this_query_contains+overlap_contain_buckets
        else:  # 添加
            add_global_bucket(query)
            # 删除root的children中属于query的
            root.delete_contains(this_contain_buckets)
            # 向query中添加bucket
            new_composed = set()
            for item in valid_contains:
                query._init_add(item)
                new_composed |= item.composed
            query.composed = new_composed
            root.composed |= query.composed
            query.volume -= contains_volume
            root.add_for_overlap(query)
            if is_close_to_zero(root.volume):
                merge_bucket_with_parent(root)

            return query, root.cover_volume-o_overlap_cover_volume, new_delta_composed
    else:
        return cret, ccover, cset


def cacl_valid_contains(this_contain_buckets: set, overlap_contain_buckets: list, query_contains: set):
    valid_contain_buckets = set()

    # 找到overlap_contain_buckets中合法的

    overlap_contain_set = set()
    for item in overlap_contain_buckets:
        if isinstance(item, set):
            overlap_contain_set |= item
        else:
            overlap_contain_set.add(item)
    overlap_contain_list = list(overlap_contain_set)
    overlap_contain_list.sort(key=lambda x: x.cover_volume)

    valid_contain_buckets |= this_contain_buckets

    for bucket in overlap_contain_list:
        valid_flag = True
        for parent in bucket.parents:
            if parent in valid_contain_buckets:
                valid_flag = False
                break
        if valid_flag:
            valid_contain_buckets.add(bucket)

    return valid_contain_buckets


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


@profile
def check_cover(query: Bucket):
    query_contains = set(global_buckets_rtree.contains(query.coordinates))
    contains_volume = np.sum(
        [global_buckets_dict[id].volume for id in query_contains])
    return query_contains, contains_volume


@profile
def feed_a_query_root(query: Bucket, root: Bucket):
    global global_buckets_rtree, global_buckets_dict

    # 检查当前query是否已经被覆盖
    query_contains, contains_volume = check_cover(query)

    # 没有被覆盖

    if not are_floats_equal(contains_volume, query.volume):
        parent = find_small_parent(query, root)  # 包含query的最小的bucket

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
        visited = set()
        find_overlap_with_query(query, parent, overlaps,
                                visited, query_contains)


def construct_histogram(queries, mins, maxs, num_tuples):
    print("Start Generate the Histogram...")
    global feature_id, global_buckets_dict
    p.dimension = len(queries[0].mins)
    root_bucket = Bucket(mins, maxs, num_tuples)
    global_buckets_dict[0] = root_bucket
    start = time.time()
    input_buckets = gen_query_buckets(queries)[:40]
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

    """ queries = [
        Bucket(mins=[0.1, 0.5], maxs=[0.6, 0.9], card=6),
        Bucket(mins=[0.4, 0.2], maxs=[0.9, 0.8], card=10),
        Bucket(mins=[0.2, 0.3], maxs=[0.7, 0.7], card=20),
    ]
    """
    hist = construct_histogram(queries, [0, 0], [1, 1], 1000)
    print(global_buckets_dict.keys())

    """ for i, c in enumerate(hist.children):
        # print("num", i)
        draw_tree(c, i) """
    draw_tree(hist, 100, global_buckets_dict)
    return hist


# test()
