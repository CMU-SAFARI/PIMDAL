import pyarrow
import pyarrow.compute as pc
import numpy as np
import time
from datetime import datetime

import pyarrow.parquet

def query6(lineitem):
    date_start = datetime.strptime('1994-01-01', '%Y-%m-%d')
    date_end = datetime.strptime('1995-01-01', '%Y-%m-%d')
    discount = 6
    quantity = 24

    sel = ((pc.field("l_shipdate") >= date_start)
        & (pc.field("l_shipdate") < date_end)
        & (pc.field("l_discount") >= 5)
        & (pc.field("l_discount") <= 7)
        & (pc.field("l_quantity") < quantity))

    lineitem_filt = lineitem.filter(sel)

    revenue = pc.multiply(lineitem_filt.column("l_extendedprice"), lineitem_filt.column("l_discount"))
    print(pc.sum(revenue).as_py() / 10000)

if __name__ == "__main__":

    sel_cols = ['l_extendedprice','l_shipdate', 'l_discount', 'l_quantity']
    lineitem = pyarrow.parquet.read_table("data/lineitem/part.0.parquet", columns=sel_cols)

    start = time.perf_counter()
    query6(lineitem)
    end = time.perf_counter()
    print("TPC-H query 6 time CPU:", end-start)