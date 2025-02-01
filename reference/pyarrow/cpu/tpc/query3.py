import pyarrow
import pyarrow.compute as pc
import numpy as np
import time
from datetime import datetime

import pyarrow.parquet

def query3(lineitem, orders, customer):
    SEGMENT = "BUILDING"
    DATE = datetime.strptime("1995-03-15", "%Y-%m-%d")
    
    l_sel = pc.field("l_shipdate") > DATE
    o_sel = pc.field("o_orderdate") < DATE
    c_sel = pc.field("c_mktsegment") == SEGMENT

    lineitem_filt = lineitem.filter(l_sel)
    orders_filt = orders.filter(o_sel)
    customer_filt = customer.filter(c_sel)

    c_o_joined = customer_filt.join(orders_filt, join_type="inner",
                                    keys="c_custkey",
                                    right_keys="o_custkey")
    f_c_joined = c_o_joined.join(lineitem_filt, join_type="inner",
                                 keys="o_orderkey",
                                 right_keys="l_orderkey")

    revenue = pc.multiply(f_c_joined.column("l_extendedprice"),
                          pc.subtract(100, f_c_joined.column("l_discount")))
    f_c_joined = f_c_joined.add_column(1, "revenue", revenue)

    result = f_c_joined.group_by(
        ["o_orderkey", "o_orderdate", "o_shippriority"]
    ).aggregate([("revenue", "sum")])
    result = result.sort_by([("revenue_sum", "descending")])
    result = result.rename_columns(["l_orderkey",
                                    "o_orderdate",
                                    "o_shippriority",
                                    "revenue"])

    print(result[:10])

if __name__ == "__main__":

    sel_cols = ["l_orderkey", "l_extendedprice", "l_discount", "l_shipdate"]
    lineitem = pyarrow.parquet.read_table("data/lineitem/part.0.parquet", columns=sel_cols)

    sel_cols = ["o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"]
    orders = pyarrow.parquet.read_table("data/orders/part.0.parquet", columns=sel_cols)

    sel_cols = ["c_mktsegment", "c_custkey"]
    customer = pyarrow.parquet.read_table("data/customer/part.0.parquet", columns=sel_cols)

    start = time.perf_counter()
    query3(lineitem, orders, customer)
    end = time.perf_counter()
    print("TPC-H query 3 time CPU:", end-start)