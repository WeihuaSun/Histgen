from pathlib import Path
import numpy as np
data_root = Path("/data")
workload_root = Path("/workload")

spj = workload_root / "sql"

imdb_schema = {
    'title': ['kind_id', 'production_year'],
    'movie_info': ["info_type_id"],
    'movie_keyword': ['keyword_id'],
    'movie_info_idx': ['info_type_id'],
    'movie_companies': ['company_id', 'company_type_id'],
    'cast_info': ['role_id', 'person_id'],
}

imdb_ranges = {
    'title': {'kind_id': (0, 7), 'production_year': (1879, 2019)},
    'movie_info': {"info_type_id": (0, 110)},
    'movie_keyword': {'keyword_id': (0, 134170)},
    'movie_info_idx': {'info_type_id': (98, 113)},
    'movie_companies': {'company_id': (0, 234997), 'company_type_id': (0, 2)},
    'cast_info': {'role_id': (0, 11), 'person_id': (0, 4061926), }
}

census_schema = {
    'a': 'integer',
    'b': 'varchar',
    'c': 'integer',
    'd': 'varchar',
    'e': 'integer',
    'f': 'varchar',
    'g': 'varchar',
    'h': 'varchar',
    'i': 'varchar',
    'j': 'varchar',
    'k': 'integer',
    'l': 'integer',
    'm': 'integer',
    'n': 'varchar',
    'o': 'varchar'
}

census_ranges = {

}


t2alias = {'title': 't', 'movie_companies': 'mc', 'cast_info': 'ci',
           'movie_info_idx': 'mi_idx', 'movie_info': 'mi', 'movie_keyword': 'mk'}


