from datetime import datetime
import pyarrow
import cudf
import rmm

import pyarrow.parquet

import time

def query5(nation, region, customer, orders, lineitem):
    DATE1 = datetime.strptime('1994-01-01', '%Y-%m-%d')
    DATE2 = datetime.strptime('1995-01-01', '%Y-%m-%d')
    REGION = 'ASIA'

    r_sel = region['r_name'] == REGION
    region_filt = region[r_sel]
    o_sel = (orders['o_orderdate'] >= DATE1) & (orders['o_orderdate'] < DATE2)
    orders_filt = orders[o_sel]

    f_n_join = region_filt.merge(nation, left_on='r_regionkey',
                            right_on='n_regionkey')
    c_f_join = f_n_join.merge(customer, left_on='n_nationkey',
                            right_on='c_nationkey')
    o_c_join = c_f_join.merge(orders_filt, left_on='c_custkey',
                            right_on='o_custkey')
    l_o_join = o_c_join.merge(lineitem, left_on='o_orderkey',
                            right_on='l_orderkey')
    joined = supplier.merge(l_o_join, left_on=['s_suppkey', 's_nationkey'],
                            right_on=['l_suppkey', 'n_nationkey'])

    joined['revenue'] = joined['l_extendedprice'] * (100 - joined['l_discount'])
    res = joined.groupby('n_name')
    res = res.agg(
        {
            'revenue' : 'sum'
        }
    ).reset_index()
    res = res.sort_values('revenue', ascending=False)

    return res.to_arrow(preserve_index=False)

if __name__ == "__main__":

    rmm.reinitialize(
        pool_allocator=True,
        managed_memory=True,
    )

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
    customer = cudf.DataFrame.from_arrow(customer)
    orders = cudf.DataFrame.from_arrow(orders)
    lineitem = cudf.DataFrame.from_arrow(lineitem)
    supplier = cudf.DataFrame.from_arrow(supplier)
    nation = cudf.DataFrame.from_arrow(nation)
    region = cudf.DataFrame.from_arrow(region)

    res = query5(nation, region, customer, orders, lineitem)
    end = time.perf_counter()

    print(res)
    print("TPC-H query 5 time GPU:", end-start)