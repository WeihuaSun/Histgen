from debug import draw_tree
import numpy as np
from rtree import index
import time


class Bucket:
    def __init__(self, mins, maxs, card=0):
        self.mins = mins
        self.maxs = maxs
        self.card = card
        self.cover_card = card
        self.children = dict()
        self.parents = dict()
        self.volume = cacl_volume(mins, maxs)
        self.density = self.card/self.volume
        self.rtree = index.Index(properties=p)
        self.contain_buckets = set()
        self.crid = 0
        self.feature_id = 0
        self.composed_set = set()
        self.constraints = []  # 该Bucket由哪些约束组成

    def add_a_child(self, bucket):
        coordinates = bucket.mins+bucket.maxs
        bucket.parents[self] = self.crid
        self.rtree.insert(self.crid, coordinates)
        self.children[self.crid] = bucket
        self.volume -= bucket.volume
        self.composed_set |= bucket.composed_set
        self.composed_set.add(bucket)
        self.crid += 1

    def add_query(self, input):
        def add(bucket):
            coordinates = bucket.mins+bucket.maxs
            bucket.parents[self] = self.crid
            self.rtree.insert(self.crid, coordinates)
            self.children[self.crid] = bucket
            delta_composed = bucket.composed_set - self.composed_set
            delta_composed.add(input)
            self.volume -= np.sum([b.volume for b in delta_composed])
            self.composed_set |= bucket.composed_set
            self.crid += 1
        if isinstance(input, list):
            for bucket in input:
                add(bucket)
        else:
            add(input)

    def add(self, input):
        if isinstance(input, list):
            for b in input:
                self.add_a_child(b)
        else:
            self.add_a_child(input)

    def delete_a_child(self, bucket, bid=None):
        if bid == None:
            bid = list(self.children.keys())[list(self.children.values()).index(bucket)]
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


class Generalized_Iterative_Scaling:
    def __init__(self) -> None:
        self.weights = []

    def cacl_entropy(self):
        pass

    def update_entropy(self):
        pass


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


def are_floats_equal(float1, float2, epsilon=1e-9):
    return abs(float1 - float2) < epsilon


def is_close_to_zero(float_num, epsilon=1e-9):
    return abs(float_num) < epsilon


def feed_a_overlap(query: Bucket, root: Bucket, feature_id: int,):
    # 处理overlap相关，这里的overlap被设计为一个查询，该查询被包含在root里，feature_id是当前处理约束的序号，用来标记已经处理过的Bucket
    parent = find_parent(query, root)
    if parent.feature_id == feature_id:
        # 这涉及到先处理了一个较大的overlap,再处理一个被包含的overlap。因为在处理大的overlap时，小的overlap已经完成处理了，所以此处略去
        # 或者两个overlap的位置大小都相同
        return []
    else:
        parent.feature_id = feature_id
        if are_coincide(parent, query):
            # TODO update constraints
            return parent
        overlaps, contains = intersect_with_children(parent, query)
        if len(overlaps) == 0 and len(contains) == 0:  # parent中没有和这个查询相交或者包含的，则将这个bucket作为parent的孩子节点
            parent.add(query)
            # TODO update constraints
            if is_close_to_zero(parent.volume):
                merge_bucket_with_parent(parent)
            return query

        # 处理contains,即query包含的一些bucket,如有必要，需要将这些bucket作为query的孩子，并且从parent中移除这些bucket
        # overlap contain的bucket不会被query分割，应该也不会被处理（？）为了避免可能的重复操作，还是更新feature_id
        cur_contains = []
        cur_contains_ids = []
        for cid in contains:
            bucket = parent.children[cid]
            cur_contains.append(bucket)
            cur_contains_ids.append(cid)
            bucket.feature_id = feature_id
            query.add_query(bucket)
            if is_close_to_zero(query.volume):
                assert len(cur_contains) == len(contains)
                return cur_contains
        assert len(cur_contains) == len(contains)

        checked_overlaps = []
        # 处理overlaps
        for oid in overlaps:
            bucket = parent.children[oid]  # bug!key error?
            if are_disjoint(bucket, query):
                continue
            # 这涉及先处理了一个较小的overlap,然后处理一个包含小overlap的大的overlap,此时小的overlap已经被处理的，不需要再次递归的处理，直接调用小的结果就行
            if bucket.feature_id == feature_id:
                continue
            else:
                overlap = get_overlap(query, bucket)  # 相交区域
                overlap.feature_id = feature_id
                ret = feed_a_overlap(overlap, bucket, feature_id)
                query.add_query(ret)
                bucket.feature_id = feature_id
                if is_close_to_zero(query.volume):
                    return cur_contains+checked_overlaps
        parent.delete_contains(cur_contains_ids, cur_contains)
        parent.add(query)
        if is_close_to_zero(parent.volume):
            merge_bucket_with_parent(parent)
        return query


def feed_a_query(query: Bucket, root: Bucket):
    """处理输入的查询,将其添加到直方图中
    Args:
        query (Bucket):输入的查询约束
        root (Bucket): 直方图的根节点

    Returns:
        _type_: _description_
    """
    feature_id = query.feature_id
    parent = find_parent(query, root)  # 找到某个Bucket,其包含query,但其字节的不包含query

    # S1:query和当前直方图中某个Bucket重合
    if are_coincide(parent, query):
        parent.cover_card = query.card
        parent.constraints.append(feature_id)
        query = parent  # 定位这个bucket
        return

    overlaps, contains = intersect_with_children(parent, query)  # 从parent中找到其直接overlap和contain的Bucket的rid

    # S2:parent中没有和这个查询相交或者包含的，则将这个bucket作为parent的孩子节点
    if len(overlaps) == 0 and len(contains) == 0:
        parent.add_a_child(query)
        if parent.volume == 0:
            merge_bucket_with_parent(parent)
        return

    # S3:处理被query包含的bucket，将这些bucket作为query的孩子，将其从parent中去除
    cur_contains = []
    cur_contains_ids = []  # 因为某个bucket的rid可能会更改(被添加到另外一个孩子中)
    for cid in contains:
        bucket = parent.children[cid]
        bucket.constraints.append(feature_id)
        cur_contains.append(bucket)
        cur_contains_ids.append(cid)
        query.add_query(bucket)
        if is_close_to_zero(query.volume):  # 如果这些包含的bucket恰好可以组成query,则不添加query
            assert len(cur_contains) == len(contains)
            return
    assert len(cur_contains) == len(contains)

    # S4:处理和query相交的bucket
    for oid in overlaps:
        bucket = parent.children[oid]
        if are_disjoint(bucket, query):  # rtree实现的overlap似乎和我们的overlap有歧义，仅仅边相交的，我们不当作overlap
            continue
        overlap = get_overlap(query, bucket)  # 相交区域
        overlap.feature_id = feature_id
        ret = feed_a_overlap(overlap, bucket, feature_id)
        query.add_query(ret)
        if is_close_to_zero(query.volume):
            return

    # 最终overlaps和contains都没有填充query，则将query添加为parent的孩子，并将parent中被query包含的孩子转换为query的孩子
    parent.delete_contains(cur_contains_ids, cur_contains)
    parent.add(query)
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


def construct_histogram(queries, mins, maxs, num_tuples):
    p.dimension = len(queries[0].mins)
    root_bucket = Bucket(mins, maxs, num_tuples)
    print("Start Generate the Histogram ")
    start = time.time()
    input_buckets = [Bucket(q.mins, q.maxs, q.card)for q in queries]
    input_buckets = deduplicate(input_buckets)  # 删除重复的Bucket
    input_buckets.sort(key=lambda x: x.density)  # 按照密度对Bucket进行排序，密度大的在最后
    num_buckets = len(input_buckets)
    #gis_solver = Generalized_Iterative_Scaling()
    for i, bucket in enumerate(input_buckets):
        if i % 10 == 0:
            print(f"Construct Histogram Step [{i+1}/{num_buckets}]")
        bucket.feature_id = i
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
    hist = construct_histogram(queries, [0, 0], [1, 1], 1000)

    for i, c in enumerate(hist.children.values()):
        print("num", i)
        draw_tree(c, i)
    draw_tree(hist, 100)
    return hist


test()
