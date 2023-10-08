from collections import OrderedDict, deque
from typing import List, Dict
from debug import draw_tree
import numpy as np
from rtree import index
import time
from line_profiler import LineProfiler

from bucket import *
from train_utils import *


p = index.Property()
p.dimension = 14
global_buckets_rtree = index.Index(properties=p)

global_queries_rtree = index.Index(properties=p)

global_buckets_dict = dict()

identifier = 1


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


#@profile
def add_global_bucket(bucket: Bucket):
    # print("add_global_bucket")
    global global_buckets_dict, global_buckets_rtree, identifier
    #global_buckets_rtree.insert(identifier, bucket.coordinates)
    global_buckets_dict[identifier] = bucket
    bucket.identifier = identifier
    bucket.composed.add(identifier)
    identifier += 1


def delete_global_bucket(bucket: Bucket):
    #print("delete_global_bucket")
    #global global_buckets_dict, global_buckets_rtree, identifier
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


def ids_2_buckets(ids):
    return set([global_buckets_dict[id] for id in ids])


def sum_volume(buckets):
    return np.sum([bucket.volume for bucket in buckets])


#@profile
def feed_a_query_naive(query: Bucket, root: Bucket, overlaps: set, input_visited: set, root_query_contain_ids: set):

    best_parent = True
    overlap_containers = []
    # Check 1:检查root的child中被query包含的，是否可以组成query
    # query_contain_child_ids = root.children & root_query_contain_ids
    # query_contain_child_buckets = ids_2_buckets(query_contain_child_ids)
    query_contain_child_buckets = set()

    # Check 2:检查root的child和query相交产生的区域
    for id in root.children.copy():
        bucket = global_buckets_dict[id]
        if id in overlaps:  # 和query相交
            if id in input_visited:  # 已经被查过,在root的parent中的其他child被检查
                if are_coincide(query, bucket.overlap_with_query.data):  # query已经被包含过
                    best_parent = False
                    cret_dataset = bucket.overlap_with_query.dataset
                    cret_composed = bucket.overlap_with_query.composed_set
                    cret_remove = bucket.overlap_with_query.to_remove_set
                else:
                    overlap_containers.append(bucket.overlap_with_query)
            else:  # 未被查过
                if are_contain(bucket, query):  # bucket包含query
                    best_parent = False
                    bucket.overlap_with_query = root.overlap_with_query
                    cret_dataset, cret_composed, cret_remove = feed_a_query_naive(
                        query, bucket, overlaps, input_visited, root_query_contain_ids)
                    bucket.overlap_with_query.dataset = cret_dataset
                    bucket.overlap_with_query.composed_set = cret_composed
                    bucket.overlap_with_query.to_remove_set = cret_remove
                else:  # bucket不包含query
                    new_overlap = get_overlap(query, bucket)
                    bucket.overlap_with_query = Container()
                    bucket.overlap_with_query.data = new_overlap
                    oret_dataset, oret_composed, oret_remove = feed_a_query_naive(
                        new_overlap, bucket, overlaps, input_visited, root_query_contain_ids)
                    bucket.overlap_with_query.dataset = oret_dataset
                    bucket.overlap_with_query.composed_set = oret_composed
                    bucket.overlap_with_query.to_remove_set = oret_remove
                    overlap_containers.append(bucket.overlap_with_query)
                input_visited.add(id)
        elif id in root_query_contain_ids:
            query_contain_child_buckets.add(bucket)
        else:  # 可能远离，相交，或者包含
            if not are_disjoint(bucket, query):
                if are_contain(query, bucket):
                    query_contain_child_buckets.add(bucket)
                    root_query_contain_ids |= bucket.composed
                else:
                    overlaps.add(id)
                    if id in input_visited:  # 已经被查过,在root的parent中的其他child被检查
                        if are_coincide(query, bucket.overlap_with_query.data):  # query已经被包含过
                            best_parent = False
                            cret_dataset = bucket.overlap_with_query.dataset
                            cret_composed = bucket.overlap_with_query.composed_set
                            cret_remove = bucket.overlap_with_query.to_remove_set
                        else:
                            overlap_containers.append(bucket.overlap_with_query)
                    else:  # 未被查过
                        if are_contain(bucket, query):  # bucket包含query
                            best_parent = False
                            bucket.overlap_with_query = root.overlap_with_query
                            cret_dataset, cret_composed, cret_remove = feed_a_query_naive(
                                query, bucket, overlaps, input_visited, root_query_contain_ids)
                            bucket.overlap_with_query.dataset = cret_dataset
                            bucket.overlap_with_query.composed_set = cret_composed
                            bucket.overlap_with_query.to_remove_set = cret_remove
                        else:  # bucket不包含query
                            new_overlap = get_overlap(query, bucket)
                            bucket.overlap_with_query = Container()
                            bucket.overlap_with_query.data = new_overlap
                            oret_dataset, oret_composed, oret_remove = feed_a_query_naive(
                                new_overlap, bucket, overlaps, input_visited, root_query_contain_ids)
                            bucket.overlap_with_query.dataset = oret_dataset
                            bucket.overlap_with_query.composed_set = oret_composed
                            bucket.overlap_with_query.to_remove_set = oret_remove
                            overlap_containers.append(bucket.overlap_with_query)
                    input_visited.add(id)
    query_contain_compose_ids = set()
    for bucket in query_contain_child_buckets:
        query_contain_compose_ids |= bucket.composed
    query_contain_compose_buckets = ids_2_buckets(query_contain_compose_ids)
    query_contain_child_volume = sum_volume(query_contain_compose_buckets)

    if are_floats_equal(query.volume, query_contain_child_volume):
        return query_contain_child_buckets, query_contain_compose_ids,set()

    if best_parent:
        overlap_contain_compose_ids = set()
        overlap_contain_buckets = set()
        to_remove_compose_ids = set()
        for item in overlap_containers:
            overlap_contain_compose_ids |= item.composed_set
            to_remove_compose_ids |= item.to_remove_set
            dataset = item.dataset
            if isinstance(dataset, set):
                overlap_contain_buckets |= dataset
            else:
                overlap_contain_buckets.add(dataset)
        query_compose_ids = overlap_contain_compose_ids | query_contain_compose_ids
        query_compose_buckets = ids_2_buckets(query_compose_ids)
        compose_volume = sum_volume(query_compose_buckets)

        valid_overlap_contain_buckets = cacl_valid_contains(overlap_contain_buckets, query_compose_ids)

        valid_contain_buckets = valid_overlap_contain_buckets | query_contain_child_buckets

        if are_floats_equal(compose_volume, query.volume):
            root.composed |= query_compose_ids
            root.composed -= to_remove_compose_ids
            return valid_overlap_contain_buckets, query_compose_ids,set()
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
                return query, query.composed, to_remove_compose_ids |{root.identifier}
            else:
                return query, query.composed, set()

    else:
        return cret_dataset, cret_composed,set()


# #@profile


def find_overlap_with_query(query: Bucket, root: Bucket, overlaps: set, input_visited: set, query_contains: set):
    target = True  # 标记，指示root的孩子节点中是否有包含query的，若target=True，则root就是query的最小父节点，将query添加到root的孩子节点中
    overlap_containers = []  # 保存root中的child和query相交产生的孩子
    this_visited = set()  # 对于此root,标记此root中已经被计算cover面积的bucket

    # 计算contains
    query_contains_child_ids = root.children & query_contains

    query_contains_child_composed = set()  # 包含在query中的，root_child的composed
    for id in query_contains_child_ids:
        child = global_buckets_dict[id]
        query_contains_child_composed |= child.composed
    ids_to_buckets = [global_buckets_dict[id] for id in query_contains_child_composed]
    contains_child_cover_volume = np.sum([bucket.volume for bucket in ids_to_buckets])

    # cacl_delta_cover and composed
    c_delta_composed = query_contains_child_composed - input_visited
    c_delta_cover = np.sum([global_buckets_dict[id].volume for id in c_delta_composed])

    query_contains_child_buckets = set([global_buckets_dict[id] for id in query_contains_child_ids])

    if are_floats_equal(contains_child_cover_volume, query.volume):
        # 更新input_visited
        input_visited |= c_delta_composed
        return query_contains_child_buckets, c_delta_cover, c_delta_composed, query_contains_child_buckets
    this_visited = query_contains_child_composed

    for id in root.children.copy():
        if id in overlaps:  # 和query相交
            bucket = global_buckets_dict[id]
            if id in input_visited:  # 已经被查过,在root的parent中的其他child被检查
                if are_coincide(query, bucket.overlap_with_query.data):  # query已经被包含过
                    target = False
                    cret_dataset = bucket.overlap_with_query.dataset
                    cret_delta_cover = 0
                    cret_delta_composed = set()
                    cret_composed = bucket.overlap_with_query.composed_set
                else:
                    this_visited |= bucket.overlap_with_query.composed_set
                    bucket.overlap_with_query.delta_composed = bucket.overlap_with_query.composed_set - input_visited
                    bucket.overlap_with_query.delta_volume = np.sum(
                        [global_buckets_dict[i] for i in bucket.overlap_with_query.delta_composed])
                    overlap_containers.append(bucket.overlap_with_query)
            else:  # 未被查过
                if are_contain(bucket, query):  # bucket包含query
                    target = False
                    bucket.overlap_with_query = root.overlap_with_query
                    cret_dataset, cret_delta_cover, cret_delta_composed, cret_composed = find_overlap_with_query(
                        query, bucket, overlaps, this_visited, query_contains)
                    bucket.overlap_with_query.dataset = cret_dataset
                    bucket.overlap_with_query.delta_cover = cret_delta_cover
                    bucket.overlap_with_query.delta_composed = cret_delta_composed
                    bucket.overlap_with_query.composed_set = cret_composed
                else:  # bucket不包含query
                    new_overlap = get_overlap(query, bucket)
                    bucket.overlap_with_query = Container()
                    bucket.overlap_with_query.data = new_overlap
                    oret_dataset, oret_delta_cover, oret_delta_composed, oret_composed = find_overlap_with_query(
                        new_overlap, bucket, overlaps, this_visited, query_contains)
                    bucket.overlap_with_query.dataset = oret_dataset
                    bucket.overlap_with_query.delta_cover = oret_delta_cover
                    bucket.overlap_with_query.delta_composed = oret_delta_composed
                    bucket.overlap_with_query.composed_set = oret_composed
                    overlap_containers.append(bucket.overlap_with_query)
                input_visited.add(id)
    if target:  # 其孩子节点均不包含input
        # 两个volume
        # 1.root剩余的volume:volume_1
        # 2.根据input_visited计算给parent增加的volume

        # 处理input_visited

        # 处理this_visited

        # 计算volume_1
        # step1:overlap_contains

        overlap_contain_buckets = set()
        overlap_cover_volume = 0
        new_query_composed_ids = set()
        for item in overlap_containers:
            overlap_cover_volume += item.delta_cover
            new_query_composed_ids |= item.delta_composed
            dataset = item.dataset
            if isinstance(dataset, set):
                overlap_contain_buckets |= dataset
            else:
                overlap_contain_buckets.add(dataset)
        query_contains |= new_query_composed_ids
        cover_volume = contains_child_cover_volume + overlap_cover_volume
        all_composed = new_query_composed_ids | query_contains_child_composed

        valid_overlap_contain_buckets = cacl_valid_contains(overlap_contain_buckets, all_composed)

        valid_contains = valid_overlap_contain_buckets | query_contains_child_buckets

        o_delta_composed = new_query_composed_ids - input_visited
        oc_delta_composed = o_delta_composed | c_delta_composed
        o_delta_cover = np.sum([global_buckets_dict[id].volume for id in o_delta_composed])
        oc_delta_cover = o_delta_cover+c_delta_cover

        if are_floats_equal(cover_volume, query.volume):
            input_visited |= oc_delta_composed
            query.composed |= all_composed
            root.composed |= query.composed
            return valid_contains, oc_delta_cover, oc_delta_composed, all_composed
        else:  # 添加
            add_global_bucket(query)
            query.composed |= all_composed
            root.composed |= query.composed
            # 删除root的children中属于query的
            root.delete_contains(query_contains_child_buckets)
            # 向query中添加bucket
            for item in valid_contains:
                query._init_add(item)
            query.volume -= cover_volume
            root.add_for_overlap(query)

            oc_delta_composed.add(query.identifier)
            oc_delta_cover += query.volume
            input_visited |= oc_delta_composed

            if is_close_to_zero(root.volume):
                merge_bucket_with_parent(root)

            return query, oc_delta_cover, oc_delta_composed, query.composed.copy()
    else:
        return cret_dataset, cret_delta_cover, cret_delta_composed, cret_composed


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
    #print("merge")
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


#@profile
def check_cover(query: Bucket):
    query_contains = set(global_buckets_rtree.contains(query.coordinates))
    contains_volume = np.sum([global_buckets_dict[id].volume for id in query_contains])
    return query_contains, contains_volume


def get_overlaps_and_contains(parent, query):
    query_contains, contains_volume = check_cover(query)
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
    return query_contains, overlaps


def get_overlaps_and_contains_naive(parent, query):
    contain_compose_ids = set()
    root_level_overlap_ids = set()
    for cid in parent.children:
        bucket = global_buckets_dict[cid]
        if are_contain(query, bucket):
            contain_compose_ids |= bucket.composed
        elif not are_disjoint(query, bucket):
            root_level_overlap_ids.add(cid)

    return contain_compose_ids, root_level_overlap_ids


#@profile
def feed_a_query_root(query: Bucket, root: Bucket):
    global global_buckets_rtree, global_buckets_dict

    parent = find_small_parent(query, root)  # 包含query的最小的bucket
    query_contains, overlaps = get_overlaps_and_contains_naive(parent, query)
    visited = set()
    feed_a_query_naive(query, parent, overlaps, visited, query_contains)
    #find_overlap_with_query(query, parent, overlaps,visited, query_contains)


def construct_histogram(queries, mins, maxs, num_tuples):
    print("Start Generate the Histogram...")
    global global_buckets_dict
    p.dimension = len(queries[0].mins)
    root_bucket = Bucket(mins, maxs, num_tuples)
    global_buckets_dict[0] = root_bucket
    start = time.time()
    input_buckets = gen_query_buckets(queries)[:50]
    gen_global_rtree(input_buckets)
    num_buckets = len(input_buckets)
    for i, bucket in enumerate(input_buckets):
        print(f"Construct Histogram Step [{i+1}/{num_buckets}]")
        feed_a_query_root(bucket, root_bucket)
    end = time.time()
    print("Generate Hisogram Time Cost:{}".format(end-start))

    for key, bucket in global_buckets_dict.items():
        if bucket.volume <= 0:
            print("volume error",bucket.volume)
    return root_bucket


def test():
    """ queries = [Bucket(mins=[0.1, 0.1], maxs=[0.6, 0.6]),
               Bucket(mins=[0.3, 0.65], maxs=[0.5, 0.7]),
               Bucket(mins=[0.3, 0.5], maxs=[0.5, 0.8]),
               Bucket(mins=[0.1, 0.1], maxs=[0.6, 0.6]), ] """

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
    ] """

    hist = construct_histogram(queries, [0, 0], [1, 1], 1000)
    print(global_buckets_dict.keys())

    """ for i, c in enumerate(hist.children):
        # print("num", i)
        draw_tree(c, i) """
    draw_tree(hist, 100, global_buckets_dict)
    return hist


# test()
