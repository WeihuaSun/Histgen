import os
#import psycopg2
import csv
import pandas as pd
import json
db_url = "postgres://postgres:1@localhost:5432/imdb19"
ban_parallel = "set max_parallel_workers_per_gather = 0;"


""" class PostgreSQL:
    def __init__(self, database_url):
        self.conn = psycopg2.connect(database_url)
        self.conn.autocommit = True
        self.cur = self.conn.cursor()

    def run(self, sql):
        self.cur.execute(sql)
        rows = self.cur.fetchall()
        return rows

    def explain(self, sql, parallel=False):
        if not parallel:
            self.cur.execute(ban_parallel)
        sql = "explain (analyse,format json)  " + sql
        result = self.run(sql)[0][0]
        return result """


def get_plan(workload_file, plan_file, db_url=db_url):
    if os.path.exists(plan_file):
        plans = pd.read_csv(plan_file)
    """ else:
        db = PostgreSQL(db_url)
        with open(workload_file, "r") as f:
            queries = f.readlines()
        with open(plan_file, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["id", "json", "sql"])
            num_query = len(queries)
            for i, query in enumerate(queries):
                plan = db.explain(query)
                writer.writerow([i, json.dumps(plan[0]), str(query[:-1])])
                print(f"Parse Plan: Step[{i} / {num_query}]")
        plans = pd.read_csv(plan_file) """
    plans = [json.loads(plan)['Plan'] for plan in plans['json']]
    return plans
