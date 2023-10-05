from collections import OrderedDict, deque
from typing import List, Dict
from debug import draw_tree
import numpy as np
from rtree import index
import time


from bucket import *
from train_utils import *


p = index.Property()

# p.dimension = 14

global_buckets_rtree = index.Index(properties=p)

global_queries_rtree = index.Index(properties=p)

global_buckets_dict = dict()

identifier = 1

feature_id = 1






def find_small_parent(query: Bucket, root: Bucket):
    stack = deque([root])
    while stack:
        node = stack.pop()
        target = True
        for id in node.children:
            bucket = global_buckets_dict[id]
            if are_contain(node, query):
                stack.append(bucket)
                target = False
                break
        if target:
            return node



def find_overlap_with_query(query: Bucket, root: Bucket, overlaps: set, input: Bucket, visited: set, new_overlaps: set):
    for id in root.children:
        if id not in visited and id in overlaps:
            bucket = global_buckets_dict[id]
            if are_contain(bucket, input):
                bucket.overlap_small_parent = root.overlap_small_parent
                bucket.overlap_small_parent[0] = bucket
                bucket.overlap_with_query = input
                find_overlap_with_query(
                    query, bucket, overlaps, input, visited, new_overlaps)
            else:
                new_overlap = get_overlap(query, bucket)
                new_overlaps.add(new_overlap)
                bucket.overlap_small_parent = [bucket]
                bucket.overlap_with_query = new_overlap
                find_overlap_with_query(
                    query, bucket, overlaps, new_overlap, visited, new_overlaps)
            visited.add(id)


def find_overlap(query: Bucket, root: Bucket, overlaps: set):
    visited = set()
    stack = deque([root])
    while stack:
        node = stack.pop()
        visited.add(node)
        overlap = get_overlap(query, node)
        node.overlap_with_query = [overlap]
        for id in node.children:
            if id in overlaps:  # overlap with query
                bucket = global_buckets_dict[id]
                if are_contain(bucket, overlap):
                    overlap_with_query = [overlap]
                    stack.append(bucket)
                else:
                    find_overlap_iter()


def find_overlap_iter(visited: set, overlaps: set,):
    pass


def feed_query_with_overlap(root: Bucket, query: Bucket, overlaps: set, visited: set):
    query_contains, contains_volume = check_cover(query)
    parent = root.overlap_small_parent[0]
    this_overlap = parent.children & overlaps
    # 被覆盖
    contains = query_contains & parent.children
    if are_floats_equal(contains_volume, query.volume):
        overlap_contains = set()
        for id in this_overlap:
            o_bucket = global_buckets_dict[id]
            overlap_contains = overlap_contains | o_bucket.children
        # 返回被覆盖的list
        return contains | overlap_contains
    # 未被覆盖
    for id in this_overlap:
        o_bucket = global_buckets_dict[id]
        overlap = o_bucket.overlap_with_query
        if isinstance(overlap,set):
            parent.add_child(overlap)
        else:
            feed_query_with_overlap(o_bucket, overlap, overlaps)


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
            parent.coordinates)) - {parent.identifier}
        intersections = set(
            global_buckets_rtree.intersection(query.coordinates))
        overlaps = parent_contains & intersections - query_contains  # 和query相交的所有bucket
        new_overlaps = set()
        find_overlap_with_query(query, parent, overlaps,
                                query, visited=set(), new_overlaps=new_overlaps)
        feed_query_with_overlap()

        visited_overlap = set()
        overlap_contains = []

        this_overlap = overlaps & parent.children
        for id in this_overlap:
            bucket = global_buckets_dict[id]
            if id in visited_overlap:
                overlap_contains.extend(bucket.overlap_with_query)
            else:
                overlap = get_overlap(bucket, query)
                overlap_contains.extend(
                    feed_iter(overlap, bucket, visited_overlap, overlaps))
                visited_overlap.add(id)


def feed_iter(overlap_query: Bucket, overlap_root: Bucket, visited_overlaps: set, overlaps: set):
    query_contains, contains_volume = check_cover(overlap_query)
    parent = find_small_parent(overlap_query, overlap_root)
    contains = query_contains & overlap_query.children
    this_overlaps = parent.children & overlaps
    this_visited = set()
    if are_floats_equal(contains_volume, overlap_query.volume):
        overlap_contains = set()
        for id in this_overlaps:
            bucket = global_buckets_dict[id]
            if id in visited:
                if isinstance(bucket.overlap_with_query, set):
                    overlap_contains = bucket.overlap_with_query | overlap_contains
                else:
                    overlap_contains.add(bucket.overlap_with_query)
            else:
                overlap = get_overlap(overlap_query, bucket)
                ret = feed_iter(overlap, bucket, visited, overlaps)

        return
    else:
        pass
        up_mark()


def find_parent(query: Bucket, root: Bucket):
    global feature_id
    visted_contain = {root.identifier}  # 包含query的遍历集合
    visited_joint = set()  # 和query相交的遍历集合
    visted_contained = set()  # 部分被query包含的集合
    visted_other = set()  # 其余
    queue = deque([root.identifier])
    while queue:
        node = queue.popleft()
        if node in visted_contain:  # node包含query，其child的可能包含，相交，或者其他（被包含或远离）
            target = True
            node_item = global_buckets_dict[node]
            for cid in node_item.children:
                if cid in visted_contain:  # 已经遍历过，且包含query
                    target = False
                    continue
                elif cid in visted_other or cid in visited_joint or cid in visted_contained:  # 已经遍历过，但不包含query
                    continue
                else:  # 没有遍历过
                    c_node_item = global_buckets_dict[cid]
                    if are_contain(c_node_item, query):  # 包含query
                        target = False
                        queue.append(cid)
                        visted_contain.add(cid)
                    elif not are_disjoint(c_node_item, query):
                        if are_contain(query, c_node_item):
                            visted_contained.add(cid)
                        else:  # overlap
                            if c_node_item.feature_father == feature_id:
                                pass
                            else:
                                queue.append(cid)
                                visited_joint.add(cid)
                    else:  # 其余
                        visted_other.add(cid)
            if target:  # 该节点的孩子节点都不包含query,则该节点是包含query的最小节点
                target_node = node_item
        else:  # node in visited_joint
            node_item = global_buckets_dict[node]
            for cid in node_item.children:
                if cid in visted_other or cid in visited_joint or cid in visted_contained:  # 已经遍历过，但不包含query
                    continue
                else:  # 没有遍历过
                    c_node_item = global_buckets_dict[cid]
                    if not are_disjoint(c_node_item, query):  # 相交
                        if are_contain(query, c_node_item):
                            visted_contained.add(cid)
                        else:
                            queue.append(cid)
                            visited_joint.add(cid)
                    else:  # 其余
                        visted_other.add(cid)
    return target_node, visited_joint, visted_contained


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
    # print("intersect_with_children")
    # 输入一个query_bucket,输出和其相交的所有的bucket
    global global_buckets_dict
    coordinates = query.coordinates
    contains = set(root.rtree.contains(coordinates)
                   ) & root.children  # query包含的bucket
    intersections = set(root.rtree.intersection(coordinates)) & root.children
    overlaps = intersections-contains
    valid_overlaps = []
    for oid in overlaps:
        bucket = global_buckets_dict[oid]
        if are_disjoint(bucket, query):  # rtree实现的overlap似乎和我们的overlap有歧义，仅仅边相交的，我们不当作overlap
            continue
        valid_overlaps.append(oid)
    return set(valid_overlaps), contains


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


def cacl_valid_overlap(overlaps, query):
    global global_buckets_dict
    tmp_rtree = index.Index(properties=p)
    tmp_overlap_buckets = []
    for oid in overlaps:
        bucket = global_buckets_dict[oid]
        overlap = get_overlap(query, bucket)  # 相交区域
        tmp_overlap_buckets.append((oid, (overlap, bucket)))
        tmp_rtree.insert(oid, overlap.coordinates)

    overlaps_sorted = OrderedDict(
        sorted(tmp_overlap_buckets, key=lambda x: x[1][0].volume))
    valid_overlaps = []
    valid_oids = set()
    while overlaps_sorted:
        oid, (overlap, bucket) = overlaps_sorted.popitem()
        valid_overlaps.append((overlap, bucket))
        valid_oids.add(oid)
        this_contains = list(tmp_rtree.contains(overlap.coordinates))
        for id in this_contains:
            if id != oid:
                (doverlap, dbucket) = overlaps_sorted.pop(id)
                tmp_rtree.delete(id, doverlap.coordinates)
    assert len(valid_oids) == len(valid_overlaps)
    return valid_overlaps


def find_checked_bucket(bucket: Bucket, visited: set):
    global global_buckets_dict, feature_id
    for cid in bucket.children:
        if cid not in visited:
            visited.add(cid)
            child = global_buckets_dict[cid]
            if child.feature_cover == feature_id:
                return child
            elif child.feature_father == feature_id:
                return find_checked_bucket(child, visited)


def feed_a_query(query: Bucket, root: Bucket):
    # print("feed")
    global global_buckets_rtree, global_buckets_dict, feature_id
    parent, visited_joint, visited_contained = find_parent(query, root)
    if are_coincide(parent, query):
        query = parent
        for cid in query.children:  # 标记child
            child = global_buckets_dict[cid]
            child.feature_cover = feature_id
        return query
    # overlaps, contains = intersect_with_children(parent, query)
    overlaps = parent.children & visited_joint
    contains = parent.children & visited_contained
    contains_buckets = set([global_buckets_dict[cid] for cid in contains])

    # parent中的孩子节点没有和query相交或者被包含的，则将这个bucket作为parent的孩子节点
    if len(overlaps) == 0 and len(contains_buckets) == 0:
        add_global_bucket(query)  # 添加全局Bucket
        parent.add_for_overlap(query)
        if is_close_to_zero(parent.volume):
            merge_bucket_with_parent(parent)
        return query

    # 处理contains,即query包含的一些bucket,如有必要，需要将这些bucket作为query的孩子，并且从parent中移除这些bucket
    # overlap contain的bucket不会被query分割，应该也不会被处理（？）为了避免可能的重复操作，还是更新feature_id
    # 根据feature_id
    if contains_buckets:
        query.add_for_contain(
            contains_buckets, global_buckets_rtree, global_buckets_dict)
        if is_close_to_zero(query.volume):  # 如果这些包含的bucket恰好可以组成query,则不添加query
            return list(contains_buckets)
        else:
            parent.delete_contains(contains_buckets)

    valid_overlaps = cacl_valid_overlap(overlaps, query)

    # 预处理overlap:
    overlap_contains_buckets = set()
    overlaps_to_remove = set()
    visited = set()
    for _, bucket in valid_overlaps:
        if bucket.feature_father == feature_id:  # 该overlap已经被处理过
            checked = find_checked_bucket(bucket, visited)
            overlaps_to_remove.add(bucket.identifier)
            if checked == None:
                continue
            overlap_contains_buckets.add(checked)
    if overlap_contains_buckets:
        query.add_for_contain(
            overlap_contains_buckets, global_buckets_rtree, global_buckets_dict)
        if is_close_to_zero(query.volume):  # 如果这些包含的bucket恰好可以组成query,则不添加query
            return list(contains_buckets | overlap_contains_buckets)

    # S4:处理和query相交的bucket
    # 有效的overlap_bucket:
    # 1.和query相交但不包含query且不被query包含
    # 2.产生的overlap区域不被其他overlap包含
    # 3.可以覆盖query的最小集合

    composed = []
    for overlap, bucket in valid_overlaps:
        if bucket.identifier not in overlaps_to_remove:
            ret = feed_a_query(overlap, bucket)
            if isinstance(ret, list):
                composed += ret
                for r in ret:
                    r.feature_cover = feature_id
            else:
                ret.feature_cover = feature_id
                composed.append(ret)
            cover_buckets = set(
                global_buckets_rtree.contains(overlap.coordinates))
            query.add_for_query(ret, cover_buckets, global_buckets_rtree,
                                global_buckets_dict)
            if is_close_to_zero(query.volume):
                update_feature(visited_joint)
                return list(contains_buckets)+composed+list(overlap_contains_buckets)

    add_global_bucket(query)
    parent.add_for_overlap(query)
    if is_close_to_zero(parent.volume):
        merge_bucket_with_parent(parent)
    update_feature(visited_joint)
    return query


def update_feature(joints):
    global global_buckets_dict, feature_id
    for bid in joints:
        bucket = global_buckets_dict[bid]
        bucket.feature_father = feature_id


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


def gen_global_rtree(buckets: List[Bucket]):
    for i, bucket in enumerate(buckets):
        global_queries_rtree.insert(i, bucket.coordinates)


def gen_query_buckets(queries):
    buckets = [Bucket(q.mins, q.maxs, q.card)for q in queries]
    unique_buckets = deduplicate(buckets)
    unique_buckets.sort(key=lambda x: x.density)
    return unique_buckets


def construct_histogram(queries, mins, maxs, num_tuples):
    # print("Start Generate the Histogram...")
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
        feed_a_query(bucket, root_bucket)

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

    queries = [
        Bucket(mins=[0.1, 0.5], maxs=[0.6, 0.9], card=6),
        Bucket(mins=[0.4, 0.2], maxs=[0.9, 0.8], card=10),
        Bucket(mins=[0.2, 0.3], maxs=[0.7, 0.7], card=20),
        # Bucket(mins=[0.3, 0.2], maxs=[0.6, 0.8], card=250),
        # Bucket(mins=[0.3, 0.2], maxs=[0.6, 0.8]),
        # Bucket(mins=[0.1, 0.1], maxs=[0.6, 0.6]),
        # Bucket(mins=[0.1, 0.65], maxs=[0.5, 0.7]),
    ]

    hist = construct_histogram(queries, [0, 0], [1, 1], 1000)
    print(global_buckets_dict.keys())

    """ for i, c in enumerate(hist.children):
        # print("num", i)
        draw_tree(c, i) """
    draw_tree(hist, 100, global_buckets_dict)
    return hist


test()
