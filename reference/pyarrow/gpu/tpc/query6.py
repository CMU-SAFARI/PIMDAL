from datetime import datetime
import pyarrow
import cudf
import rmm

import pyarrow.parquet

import time

def query6(lineitem):
    date_start = datetime.strptime('1994-01-01', '%Y-%m-%d')
    date_end = datetime.strptime('1995-01-01', '%Y-%m-%d')
    discount = 6
    quantity = 24

    sel = ((lineitem['l_shipdate'] >= date_start)
        & (lineitem['l_shipdate'] < date_end)
        & (lineitem['l_discount'] >= 5)
        & (lineitem['l_discount'] <= 7)
        & (lineitem['l_quantity'] < quantity))

    lineitem_filt = lineitem[sel]

    revenue = lineitem_filt['l_extendedprice'] * lineitem['l_discount']

    return revenue.sum()/10000

if __name__ == "__main__":

    rmm.reinitialize(
        pool_allocator=True,
        managed_memory=True,
    )

    sel_cols = ['l_extendedprice','l_shipdate', 'l_discount', 'l_quantity']
    lineitem = pyarrow.parquet.read_table("data/lineitem/part.0.parquet", columns=sel_cols)

    start = time.perf_counter()
    lineitem = cudf.DataFrame.from_arrow(lineitem)

    res = query6(lineitem)
    end = time.perf_counter()

    print("Revenue", res)
    print("TPC-H query 6 time GPU:", end-start)