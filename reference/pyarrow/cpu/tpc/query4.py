import pyarrow
import pyarrow.compute as pc
import numpy as np
import time
from datetime import datetime

import pyarrow.parquet

def query4(orders, lineitem):
    date_start = datetime.strptime('1993-07-01', '%Y-%m-%d')
    date_end = datetime.strptime('1993-10-01', '%Y-%m-%d')
    
    l_sel = pc.field("l_commitdate") < pc.field("l_receiptdate")
    o_sel = (pc.field('o_orderdate') >= date_start) & (pc.field("o_orderdate") < date_end)

    lineitem_filt = lineitem.filter(l_sel)
    order_filt = orders.filter(o_sel)
    order_filt = order_filt.select(["o_orderkey", "o_orderpriority"])

    joined = order_filt.join(lineitem_filt, join_type="inner",
                             keys='o_orderkey',
                             right_keys='l_orderkey')

    joined = joined.select(['o_orderpriority', 'o_orderkey'])
    result = joined.group_by("o_orderpriority").aggregate([("o_orderkey", "count_distinct")])
    result = result.sort_by([("o_orderpriority", "ascending")])

    print(result)

if __name__ == "__main__":

    sel_cols = ['l_orderkey', 'l_commitdate','l_receiptdate']
    lineitem = pyarrow.parquet.read_table("data/lineitem/part.0.parquet", columns=sel_cols)

    sel_cols = ['o_orderkey', 'o_orderdate', 'o_orderpriority']
    orders = pyarrow.parquet.read_table("data/orders/part.0.parquet", columns=sel_cols)

    start = time.perf_counter()
    query4(orders, lineitem)
    end = time.perf_counter()
    print("TPC-H query 4 time CPU:", end-start)