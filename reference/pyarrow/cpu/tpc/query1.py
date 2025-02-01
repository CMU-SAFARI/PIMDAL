import pyarrow
import pyarrow.compute as pc
import numpy as np
import time
from datetime import datetime

import pyarrow.parquet

def query1(lineitem):
    date = datetime.strptime("1998-9-02", "%Y-%m-%d")
    l_sel = pc.field("l_shipdate") < date

    lineitem_filt = lineitem.filter(l_sel)
    sum_qty = lineitem_filt.column("l_quantity")
    sum_base_price = lineitem_filt.column("l_extendedprice")
    sum_disc_price = pc.multiply(lineitem_filt.column("l_extendedprice"),
                                 pc.subtract(100, lineitem_filt.column("l_discount")))
    sum_charge = pc.multiply(pc.multiply(lineitem_filt.column("l_extendedprice"),
                                         pc.subtract(100, lineitem_filt.column("l_discount"))),
                             pc.add(100, lineitem_filt.column("l_tax")))
    
    lineitem_filt = lineitem_filt.add_column(1, "sum_qty", sum_qty)
    lineitem_filt = lineitem_filt.add_column(2, "sum_base_price", sum_base_price)
    lineitem_filt = lineitem_filt.add_column(3, "sum_disc_price", sum_disc_price)
    lineitem_filt = lineitem_filt.add_column(4, "sum_charge", sum_charge)

    lineitem_grouped = lineitem_filt.group_by(["l_returnflag", "l_linestatus"])
    lineitem_grouped = lineitem_grouped.aggregate(
        [
            ("sum_qty", "sum"),
            ("sum_base_price", "sum"),
            ("sum_disc_price", "sum"),
            ("sum_charge", "sum"),
            ("l_quantity", "mean"),
            ("l_extendedprice", "mean"),
            ("l_discount", "mean"),
            ("l_orderkey", "count")
        ]
    )

    result = lineitem_grouped.sort_by([("l_returnflag", "ascending"),
                                       ("l_linestatus", "ascending")])
    result = result.rename_columns(["l_returnflag",
                                    "l_linestatus",
                                    "sum_qty",
                                    "sum_base_price",
                                    "sum_disc_price",
                                    "sum_charge",
                                    "avg_qty",
                                    "avg_price",
                                    "avg_disc",
                                    "count_order"])

    print(result)


if __name__ == "__main__":
    sel_cols = ["l_orderkey", "l_returnflag", "l_linestatus", "l_quantity",
                "l_tax", "l_extendedprice", "l_discount", "l_shipdate"]
    lineitem = pyarrow.parquet.read_table("data/lineitem/part.0.parquet", columns=sel_cols)

    start = time.perf_counter()
    query1(lineitem)
    end = time.perf_counter()
    print("TPC-H query 1 time CPU:", end-start)