from datetime import datetime
import pyarrow
import cudf
import rmm

import pyarrow.parquet

import time

def query1(lineitem):
    date = datetime.strptime('1998-9-02', '%Y-%m-%d')
    l_sel = (lineitem['l_shipdate'] < date)

    lineitem_filt = lineitem[l_sel]
    lineitem_filt['sum_qty'] = lineitem_filt['l_quantity']
    lineitem_filt['sum_base_price'] = lineitem_filt['l_extendedprice']
    lineitem_filt['sum_disc_price'] = (lineitem_filt['l_extendedprice'] *
                                    (100-lineitem_filt['l_discount']))
    lineitem_filt['sum_charge'] = (lineitem_filt['l_extendedprice'] *
                                (100-lineitem_filt['l_discount']) *
                                (100+lineitem_filt['l_tax']))
    lineitem_filt['avg_qty'] = (lineitem_filt['l_quantity'])
    lineitem_filt['avg_price'] = (lineitem_filt['l_extendedprice'])
    lineitem_filt['avg_disc'] = (lineitem_filt['l_discount'])
    lineitem_filt['count_order'] = lineitem_filt['l_orderkey']

    lineitem_grouped = lineitem_filt.groupby(['l_returnflag', 'l_linestatus'])
    lineitem_grouped = lineitem_grouped.agg(
        {
            'sum_qty' : 'sum',
            'sum_base_price' : 'sum',
            'sum_disc_price' : 'sum',
            'sum_charge' : 'sum',
            'avg_qty' : 'mean',
            'avg_price' : 'mean',
            'avg_disc' : 'mean',
            'count_order' : 'count'
        }
    ).reset_index()

    res = lineitem_grouped.sort_values(['l_returnflag', 'l_linestatus'])
    
    return res.to_arrow(preserve_index=False)

if __name__ == "__main__":

    rmm.reinitialize(
        pool_allocator=True,
        managed_memory=True,
    )

    sel_cols = ['l_orderkey', 'l_returnflag', 'l_linestatus', 'l_quantity',
            'l_tax', 'l_extendedprice', 'l_discount', 'l_shipdate']
    lineitem = pyarrow.parquet.read_table("data/lineitem/part.0.parquet", columns=sel_cols)

    start = time.perf_counter()
    lineitem = cudf.DataFrame.from_arrow(lineitem)
    
    res = query1(lineitem)
    end = time.perf_counter()

    print(res)
    print("TPC-H query 1 time GPU:", end-start)