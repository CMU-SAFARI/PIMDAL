from datetime import datetime
import pyarrow
import cudf
import rmm

import pyarrow.parquet

import time

def query3(lineitem, orders, customer):
    SEGMENT = 'BUILDING'
    DATE = datetime.strptime('1995-03-15', '%Y-%m-%d')

    l_sel = lineitem['l_shipdate'] > DATE
    o_sel = orders['o_orderdate'] < DATE
    c_sel = customer['c_mktsegment'] == SEGMENT

    lineitem_filt = lineitem[l_sel]
    orders_filt = orders[o_sel]
    customer_filt = customer[c_sel]

    c_o_joined = customer_filt.merge(orders_filt, left_on='c_custkey',
                                    right_on='o_custkey')
    f_c_joined = c_o_joined.merge(lineitem_filt, left_on='o_orderkey',
                                right_on='l_orderkey')

    f_c_joined['revenue'] = (f_c_joined['l_extendedprice'] *
                            (100 - f_c_joined['l_discount']))

    f_c_grouped = f_c_joined.groupby(
        ['l_orderkey', 'o_orderdate', 'o_shippriority']
    )
    f_c_grouped = f_c_grouped.agg(
        {
            'revenue' : 'sum'
        }
    ).reset_index()

    res = f_c_grouped.sort_values(['revenue'], ascending=False)
    res = res.loc[:, ['l_orderkey', 'revenue', 'o_orderdate', 'o_shippriority']]
    
    return res.to_arrow(preserve_index=False)

if __name__ == "__main__":

    rmm.reinitialize(
        pool_allocator=True,
        managed_memory=True,
    )

    sel_cols = ['l_orderkey', 'l_extendedprice', 'l_discount', 'l_shipdate']
    lineitem = pyarrow.parquet.read_table("data/lineitem/part.0.parquet", columns=sel_cols)

    sel_cols = ['c_mktsegment', 'c_custkey']
    customer = pyarrow.parquet.read_table("data/customer/part.0.parquet", columns=sel_cols)

    sel_cols = ['o_orderkey', 'o_custkey', 'o_orderdate', 'o_shippriority']
    orders = pyarrow.parquet.read_table("data/orders/part.0.parquet", columns=sel_cols)

    start = time.perf_counter()
    lineitem = cudf.DataFrame.from_arrow(lineitem)
    customer = cudf.DataFrame.from_arrow(customer)
    orders = cudf.DataFrame.from_arrow(orders)

    res = query3(lineitem, orders, customer)
    end = time.perf_counter()

    print(res)
    print("TPC-H query 3 time GPU:", end-start)