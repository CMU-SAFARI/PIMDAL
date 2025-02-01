import pyarrow
import pyarrow.compute as pc
import numpy as np
import time
from datetime import datetime

import pyarrow.parquet

def query5(nation, region, customer, orders, lineitem, supplier):
    DATE1 = datetime.strptime('1994-01-01', '%Y-%m-%d')
    DATE2 = datetime.strptime('1995-01-01', '%Y-%m-%d')
    REGION = 'ASIA'
    
    r_sel = pc.field("r_name") == REGION
    region_filt = region.filter(r_sel)

    o_sel = (pc.field("o_orderdate") >= DATE1) & (pc.field("o_orderdate") < DATE2)
    orders_filt = orders.filter(o_sel)

    f_n_join = region_filt.join(nation, join_type='inner',
                                keys='r_regionkey',
                                right_keys='n_regionkey')
    c_f_join = f_n_join.join(customer, join_type='inner',
                             keys='n_nationkey',
                             right_keys='c_nationkey')
    o_c_join = c_f_join.join(orders_filt, join_type='inner',
                             keys='c_custkey',
                             right_keys='o_custkey')
    l_o_join = o_c_join.join(lineitem, join_type='inner',
                             keys='o_orderkey',
                             right_keys='l_orderkey')
    joined = supplier.join(l_o_join, join_type='inner',
                           keys=['s_suppkey', 's_nationkey'],
                           right_keys=['l_suppkey', 'n_nationkey'])
    
    revenue = pc.multiply(joined.column('l_extendedprice'),
                          pc.subtract(100, joined.column('l_discount')))
    
    joined = joined.add_column(1, "revenue", revenue)
    joined = joined.group_by("n_name").aggregate([("revenue", "sum")])

    res = joined.sort_by([("revenue_sum", "descending")])
    print(res)

if __name__ == "__main__":

    sel_cols = ['c_custkey', 'c_nationkey']
    customer = pyarrow.parquet.read_table("data/customer/part.0.parquet", columns=sel_cols)

    sel_cols = ['o_custkey', 'o_orderkey', 'o_orderdate']
    orders = pyarrow.parquet.read_table("data/orders/part.0.parquet", columns=sel_cols)

    sel_cols = ['l_orderkey', 'l_suppkey', 'l_extendedprice', 'l_discount']
    lineitem = pyarrow.parquet.read_table("data/lineitem/part.0.parquet", columns=sel_cols)

    sel_cols = ['s_suppkey', 's_nationkey']
    supplier = pyarrow.parquet.read_table("data/supplier/part.0.parquet", columns=sel_cols)

    sel_cols = ['n_name', 'n_nationkey', 'n_regionkey']
    nation = pyarrow.parquet.read_table("data/nation/part.0.parquet", columns=sel_cols)

    sel_cols = ['r_name', 'r_regionkey']
    region = pyarrow.parquet.read_table("data/region/part.0.parquet", columns=sel_cols)

    start = time.perf_counter()
    query5(nation, region, customer, orders, lineitem, supplier)
    end = time.perf_counter()
    print("TPC-H query 5 time CPU:", end-start)