# https://github.com/pola-rs/tpch/blob/main/dask_queries/q4.py

from datetime import datetime
import pyarrow
import cudf
import rmm

import pyarrow.parquet

import time

def query4(lineitem, orders):
    date_start = datetime.strptime('1993-07-01', '%Y-%m-%d')
    date_end = datetime.strptime('1993-10-01', '%Y-%m-%d')

    l_sel = lineitem['l_commitdate'] < lineitem['l_receiptdate']
    o_sel = (orders['o_orderdate'] >= date_start) & (orders['o_orderdate'] < date_end)

    lineitem_filt = lineitem[l_sel]
    order_filt = orders[o_sel]
    order_filt = order_filt[['o_orderkey', 'o_orderpriority']]

    joined = order_filt.merge(lineitem_filt, left_on='o_orderkey',
                            right_on='l_orderkey')
    joined = joined.drop_duplicates(subset=['o_orderkey'])
    joined = joined[['o_orderpriority', 'o_orderkey']]

    result = joined.groupby('o_orderpriority')['o_orderkey']
    result = result.count().reset_index().sort_values(['o_orderpriority'])
    result = result.rename(columns={'o_orderkey' : 'order_count'})

    return result.to_arrow(preserve_index=False)

if __name__ == "__main__":

    rmm.reinitialize(
        pool_allocator=True,
        managed_memory=True,
    )

    sel_cols = ['l_orderkey', 'l_commitdate','l_receiptdate']
    lineitem = pyarrow.parquet.read_table("data/lineitem/part.0.parquet", columns=sel_cols)

    sel_cols = ['o_orderkey', 'o_orderdate', 'o_orderpriority']
    orders = pyarrow.parquet.read_table("data/orders/part.0.parquet", columns=sel_cols)

    start = time.perf_counter()
    lineitem = cudf.DataFrame.from_arrow(lineitem)
    orders = cudf.DataFrame.from_arrow(orders)

    res = query4(lineitem, orders)
    end = time.perf_counter()

    print(res)
    print("TPC-H query 4 time GPU:", end-start)