import argparse
import pandas as pd
import json
from line_profiler import LineProfiler
from dataset_utils import parse_plan_single
from pathlib import Path

import constants
from queryplan import get_plan
from hist_rtree import construct_histogram


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default='census',
                        help='Dataset.')
    parser.add_argument('--workload-size',
                        type=int,
                        default=20000,
                        help='Number of queries to train for.')
    args = parser.parse_args()
    data_root = Path("./data")
    query_file = data_root / f"{args.dataset}_query.sql"
    plan_file = data_root / f"{args.dataset}_plan.csv"
    if args.dataset == "census":
        table_info = data_root/"census.json"
        db_url = "postgres://postgres:1@localhost:5432/census"
        single = True
    elif args.dataset == "dmv":
        table_info = data_root/"dmv.json"
        db_url = "postgres://postgres:1@localhost:5432/dmv"
        single = True
    else:
        db_url = "postgres://postgres:1@localhost:5432/imdb19"
        single = False
    plans = get_plan(query_file, plan_file, db_url)

    if single:
        constraints, table = parse_plan_single(plans, table_info)
        mins = [col.min_index for col in table.columns.values()]
        maxs = [col.max_index+1 for col in table.columns.values()]
        #hist = construct(constraints,mins,maxs,table.cardinality)
        rtree = construct_histogram(constraints, mins, maxs, num_tuples=48842)
    else:
        pass
        """ constraints = parse_plan_multi(
            plans, constants.imdb_ranges, constants.imdb_schema) """

    #hist = construct(constraints.filter['cast_info'], [0, 0], [11, 4061926], 1000)
