import time
import json

class Table(object):

    def __init__(self, info_dict):
        self.name = info_dict["name"]
        self.cardinality = info_dict["cardinality"]
        self.columns = self._build_columns(info_dict['columns'])
        self.cols_name = list(info_dict['columns'].keys())
        self.value_to_index = {c: i for i, c in enumerate(self.cols_name)}

    def _build_columns(self, columns_dict):
        cols = dict()
        for c, ci in columns_dict.items():
            cols[c] = Column(c, ci)
        return cols

    def column_value_to_index(self, col, val):
        return self.columns[col].value_to_index[val]


class Column(object):
    def __init__(self, name, info_dict):
        self.name = name
        self.size = info_dict['size']
        self.distinct_values = list(info_dict['distinct_value'].keys())
        self.distinct_values_with_counts = info_dict['distinct_value']
        self.value_to_index = {v: i for i,
                               v in enumerate(self.distinct_values)}
        self.index_to_value = self.distinct_values
        self.min_val = self.distinct_values[0]
        self.max_val = self.distinct_values[-1]
        self.min_index = 0
        self.max_index = len(self.distinct_values)-1


class Constraint:
    def __init__(self):
        self.filter = dict()
        self.join = []
        self.join_tree = []
        self.norm_filter = []

    def append(self, node):
        if isinstance(node, FilterNode):
            if node.table in self.filter:
                self.filter[node.table].append(node)
            else:
                self.filter[node.table] = [node]

        if isinstance(node, JoinNode):
            self.join.append(node)

    def append_join_tree(self, node):
        self.join_tree.append(node)


class FilterNode(object):
    def __init__(self, table, cond, card):
        self.table = table
        self.cond = cond
        self.card = card
        self.mins = []
        self.maxs = []


class JoinNode(object):
    def __init__(self, card, left=None, right=None, cond=None):
        self.left = left
        self.right = right
        self.cond = cond
        self.card = card
        if self.left is None:
            self.type = 'leaf'
        else:
            self.type = 'inner'


class LeafNode(object):
    def __init__(self, table, cond=None):
        self.table = table
        self.cond = cond


class Operator:
    """
    实现了将部分操作符转换为约束
    """

    def __init__(self):
        pass

    def step(self, plan, children, constraint):
        name = plan['Node Type']
        assert name in ['Sort', 'Bitmap Index Scan', 'BitmapAnd', 'Index Scan', 'Nested Loop',
                        'Materialize', 'Merge Join', 'Bitmap Heap Scan', 'Seq Scan', 'Hash', 'Hash Join']
        name = "_".join(name.split(" ")).lower()
        return getattr(self, name)(plan, children, constraint)

    # 顺序扫描Seq Scan
    def seq_scan(self, plan, children, constraint):
        cond = []
        table = plan['Relation Name']
        card = plan['Actual Rows']
        if 'Filter' in plan:
            # eg. ((person_id < 1931896) AND (role_id > 1))
            cond_str = plan['Filter']
            if 'AND' in cond_str:
                cond = cond_str[2:-2].split(') AND (')
            else:
                cond.append(cond_str[1:-1])
        node = FilterNode(table, cond, card)
        if cond:
            constraint.append(node)
        return node

    # 索引扫描Index Scan
    def index_scan(self, plan, children, constraint):
        table = plan['Relation Name']
        cond = []
        if 'Index Cond' in plan:
            cond_str = plan['Index Cond']
            card = plan['Actual Rows']
            if cond_str[-2].isnumeric():
                cond.append(cond_str[1:-1])
                if 'Filter' in plan:
                    remove_by_filter = plan['Rows Removed by Filter']
                    node = FilterNode(table, cond, card+remove_by_filter)
                    constraint.append(node)
                    # eg. ((person_id < 1931896) AND (role_id > 1))
                    cond_str = plan['Filter']
                    if 'AND' in cond_str:
                        filter_cond = cond_str[2:-2].split(') AND (')
                    else:
                        filter_cond = [cond_str[1:-1]]
                    cond = cond[:] + filter_cond
                node = FilterNode(table, cond, card)
                constraint.append(node)
            else:  # Nested Loop Join condition
                node = LeafNode(table, cond)
        else:  # Other Join condition
            node = LeafNode(table)
        return node

    # 位图哈希扫描
    def bitmap_heap_scan(self, plan, child, constraint):
        card = plan['Actual Rows']
        table = plan['Relation Name']
        cond_str = plan['Recheck Cond']
        if 'AND' in cond_str:
            cond = cond_str[2:-2].split(') AND (')
        else:
            cond = [cond_str[1:-1]]
        if 'Filter' in plan:
            remove_by_filter = plan['Rows Removed by Filter']
            node = FilterNode(table, cond, card+remove_by_filter)
            constraint.append(node)
            # eg. ((person_id < 1931896) AND (role_id > 1))
            cond_str = plan['Filter']
            if 'AND' in cond_str:
                filter_cond = cond_str[2:-2].split(') AND (')
            else:
                filter_cond = [cond_str[1:-1]]
            cond = cond[:] + filter_cond
        node = FilterNode(table, cond, card)
        constraint.append(node)
        return node

    # 位图索引扫描
    def bitmap_index_scan(self, plan, child, constraint):
        card = plan['Actual Rows']
        # table = plan['Relation Name']
        # eg. ((person_id < 1931896) AND (role_id > 1))
        cond_str = plan['Index Cond']
        if 'AND' in cond_str:
            cond = cond_str[2:-2].split(') AND (')
        else:
            cond = [cond_str[1:-1]]
        node = FilterNode(None, cond, card)
        # constraint.append(node)
        return node

    # 嵌套循环连接
    def nested_loop(self, plan, child, constraint):
        card = plan['Actual Rows']
        node = JoinNode(card, child[0], child[1], child[1].cond)
        constraint.append(node)
        return node

    # 归并连接
    def merge_join(self, plan, child, constraint):
        card = plan['Actual Rows']
        cond = plan['Merge Cond']
        node = JoinNode(card, child[0], child[1], cond)
        constraint.append(node)
        return node

    # 哈希连接
    def hash_join(self, plan, child, constraint):
        card = plan['Actual Rows']
        cond = plan['Hash Cond']
        node = JoinNode(card, child[0], child[1], cond)
        constraint.append(node)
        return node

    # 排序
    def sort(self, plan, child, constraint):
        # child[0].card = plan['Actual Rows']
        return child[0]

    # 哈希
    def hash(self, plan, child, constraint):
        # child[0].card = plan['Actual Rows']
        return child[0]

    #
    def gather_merge(self, plan, child, constraint):
        # child[0].card = plan['Actual Rows']
        return child[0]

    #
    def gather(self, plan, child, constraint):
        # child[0].card = plan['Actual Rows']
        return child[0]

    # 物化
    def materialize(self, plan, child, constraint):
        # child[0].card = plan['Actual Rows']
        return child[0]

    # 位图和
    def bitmapand(self, plan, child, constraint):
        return None

def parse_cond(cond_str):
    [col, op, val] = cond_str.split(r' ')
    if '::text' in col:
        #example:"(n)::text >= 'United-States'::text"
        col = col[1:-7]
        val = val[1:-7]
    return [col, op, val]

def package_filter_single(filters, table_info: Table, norm=False):
    #filters_have_checked = []
    filters_ro_return = []
    mins = [col.min_index for col in table_info.columns.values()]
    maxs = [col.max_index+1 for col in table_info.columns.values()]  # +1开区间
    for f in filters:
        f.mins = mins[:]
        f.maxs = maxs[:]
        if fill(f, table_info, norm):  # and f.cond not in filters_have_checked
            # filters_have_checked.append(f.cond)
            filters_ro_return.append(f)
    return filters_ro_return

def fill(filter: FilterNode, table: Table, norm: bool):
    # [a,b)
    mins = filter.mins
    maxs = filter.maxs
    conds = filter.cond[:]
    filter.cond = []
    for cond in conds:
        cond_re = [col, op, val] = parse_cond(cond)
        filter.cond.append(cond_re)
        val = table.column_value_to_index(col, val)
        idx = table.value_to_index[col]
        # min_val is min value but max_val is max value+1
        min_val, max_val = mins[idx], maxs[idx]
        if op == "=":  # [val,val+1)
            if val < min_val or val >= max_val:
                return False
            else:
                mins[idx] = val
                maxs[idx] = val+1
        elif op == ">":  # [val+1,max_val)
            if val < min_val:
                continue
            elif val >= max_val-1:
                return False
            else:
                mins[idx] = val+1
                maxs[idx] = max_val
        elif op == ">=":  # [val,max_val)
            if val <= min_val:
                continue
            elif val >= max_val:
                return False
            else:
                mins[idx] = val
                maxs[idx] = max_val
        elif op == "<":  # [min_val,val)
            if val >= max_val:
                continue
            elif val <= min_val:
                return False
            else:
                mins[idx] = min_val
                maxs[idx] = val
        elif op == "<=":  # [min_val,val+1)
            if val >= max_val-1:
                continue
            elif val < min_val:
                return False
            else:
                mins[idx] = min_val
                maxs[idx] = val+1
    # filter.cond.sort()
    return True

def traverse_plan(plan, constraint):
    children = []
    if 'Plans' in plan:
        for child in plan['Plans']:
            ret = traverse_plan(child, constraint)
            children.append(ret)
    ret = Operator().step(plan, children, constraint)
    return ret

def load_single_table_info(json_file):
    with open(json_file, "r") as f:
        table_info = json.load(f)
    table = Table(table_info)
    return table


def parse_plan_single(plans, table):
    print("Parse Plans...")
    start = time.time()
    table_info = load_single_table_info(table)
    constraint = []
    for plan in plans:
        traverse_plan(plan, constraint)
    package_filter_single(constraint, table_info)
    end = time.time()
    print("Done. Time Cost:{}".format(end-start))
    return constraint, table_info
