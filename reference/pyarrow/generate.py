import os
import duckdb
import pyarrow
import pyarrow.parquet
import pyarrow.compute as pc

import pandas as pd

scale_factor = 1
chunk_size = 1000000

def write(reader, schema, name):
    path = "data/" + name
    os.makedirs(path, exist_ok=True)

    writer = pyarrow.parquet.ParquetWriter(path + "/part.0.parquet", schema)

    while True:
        try:
            batch = reader.read_next_batch().cast(target_schema=schema)
            writer.write_batch(batch)
        except StopIteration:
            print("Finished generating {}.".format(name))
            break

    writer.close()

def gen_part():
    duckdb.sql("UPDATE part SET p_retailprice=p_retailprice*100")
    part = duckdb.sql("SELECT * FROM part").fetch_arrow_reader(chunk_size)

    schema = pyarrow.schema([
        pyarrow.field('p_partkey', pyarrow.uint32()),
        pyarrow.field('p_name', pyarrow.string()),
        pyarrow.field('p_mfgr', pyarrow.string()),
        pyarrow.field('p_brand', pyarrow.string()),
        pyarrow.field('p_type', pyarrow.string()),
        pyarrow.field('p_size', pyarrow.int32()),
        pyarrow.field('p_container', pyarrow.string()),
        pyarrow.field('p_retailprice', pyarrow.int64()),
        pyarrow.field('p_comment', pyarrow.string())
    ])

    write(part, schema, "part")

def gen_supplier():
    duckdb.sql("UPDATE supplier SET s_acctbal=s_acctbal*100")
    supplier = duckdb.sql("SELECT * FROM supplier").fetch_arrow_reader(chunk_size)

    schema = pyarrow.schema([
        pyarrow.field('s_suppkey', pyarrow.uint32()),
        pyarrow.field('s_name', pyarrow.string()),
        pyarrow.field('s_address', pyarrow.string()),
        pyarrow.field('s_nationkey', pyarrow.uint32()),
        pyarrow.field('s_phone', pyarrow.string()),
        pyarrow.field('s_acctbal', pyarrow.int64()),
        pyarrow.field('s_comment', pyarrow.string())
    ])

    write(supplier, schema, "supplier")

def gen_partsupp():
    duckdb.sql("UPDATE partsupp SET ps_supplycost=ps_supplycost*100")
    partsupp = duckdb.sql("SELECT * FROM partsupp").fetch_arrow_reader(chunk_size)

    schema = pyarrow.schema([
        pyarrow.field('ps_partkey', pyarrow.uint32()),
        pyarrow.field('ps_suppkey', pyarrow.uint32()),
        pyarrow.field('ps_availqty', pyarrow.int32()),
        pyarrow.field('ps_supplycost', pyarrow.int64()),
        pyarrow.field('ps_comment', pyarrow.string())
    ])

    write(partsupp, schema, "partsupp")

def gen_customer():
    duckdb.sql("UPDATE customer SET c_acctbal=c_acctbal*100")
    customer = duckdb.sql("SELECT * FROM customer").fetch_arrow_reader(chunk_size)

    schema = pyarrow.schema([
        pyarrow.field('c_custkey', pyarrow.uint32()),
        pyarrow.field('c_name', pyarrow.string()),
        pyarrow.field('c_address', pyarrow.string()),
        pyarrow.field('c_nationkey', pyarrow.uint32()),
        pyarrow.field('c_phone', pyarrow.string()),
        pyarrow.field('c_acctbal', pyarrow.int64()),
        pyarrow.field('c_mktsegment', pyarrow.string()),
        pyarrow.field('c_comment', pyarrow.string())
    ])

    write(customer, schema, "customer")

def gen_orders():
    duckdb.sql("UPDATE orders SET o_totalprice=o_totalprice*100")
    orders = duckdb.sql("SELECT * FROM orders").fetch_arrow_reader(chunk_size)

    schema = pyarrow.schema([
        pyarrow.field('o_orderkey', pyarrow.uint32()),
        pyarrow.field('o_custkey', pyarrow.uint32()),
        pyarrow.field('o_orderstatus', pyarrow.string()),
        pyarrow.field('o_totalprice', pyarrow.int64()),
        pyarrow.field('o_orderdate', pyarrow.date32()),
        pyarrow.field('o_orderpriority', pyarrow.string()),
        pyarrow.field('o_clerk', pyarrow.string()),
        pyarrow.field('o_shippriority', pyarrow.int32()),
        pyarrow.field('o_comment', pyarrow.string())
    ])

    write(orders, schema, "orders")

def gen_lineitem():
    duckdb.sql("UPDATE lineitem SET l_extendedprice=l_extendedprice*100")
    duckdb.sql("UPDATE lineitem SET l_discount=l_discount*100")
    duckdb.sql("UPDATE lineitem SET l_tax=l_tax*100")
    lineitem = duckdb.sql("SELECT * FROM lineitem").fetch_arrow_reader(chunk_size)

    schema = pyarrow.schema([
        pyarrow.field('l_orderkey', pyarrow.uint32()),
        pyarrow.field('l_partkey', pyarrow.uint32()),
        pyarrow.field('l_suppkey', pyarrow.uint32()),
        pyarrow.field('l_linenumber', pyarrow.int32()),
        pyarrow.field('l_quantity', pyarrow.int64()),
        pyarrow.field('l_extendedprice', pyarrow.int64()),
        pyarrow.field('l_discount', pyarrow.int64()),
        pyarrow.field('l_tax', pyarrow.int64()),
        pyarrow.field('l_returnflag', pyarrow.string()),
        pyarrow.field('l_linestatus', pyarrow.string()),
        pyarrow.field('l_shipdate', pyarrow.date32()),
        pyarrow.field('l_commitdate', pyarrow.date32()),
        pyarrow.field('l_receiptdate', pyarrow.date32()),
        pyarrow.field('l_shipinstruct', pyarrow.string()),
        pyarrow.field('l_shipmode', pyarrow.string()),
        pyarrow.field('l_comment', pyarrow.string()),
    ])

    write(lineitem, schema, "lineitem")

def gen_nation():
    nation = duckdb.sql("SELECT * FROM nation").fetch_arrow_reader(chunk_size)

    schema = pyarrow.schema([
        pyarrow.field('n_nationkey', pyarrow.uint32()),
        pyarrow.field('n_name', pyarrow.string()),
        pyarrow.field('n_regionkey', pyarrow.uint32()),
        pyarrow.field('n_comment', pyarrow.string())
    ])

    write(nation, schema, "nation")

def gen_region():
    region = duckdb.sql("SELECT * FROM region").fetch_arrow_reader(chunk_size)

    schema = pyarrow.schema([
        pyarrow.field('r_regionkey', pyarrow.uint32()),
        pyarrow.field('r_name', pyarrow.string()),
        pyarrow.field('r_comment', pyarrow.string())
    ])

    write(region, schema, "region")

duckdb.sql("CALL dbgen(sf = {});".format(scale_factor))
print("Data generation finished.")

gen_part()
gen_supplier()
gen_partsupp()
gen_customer()
gen_orders()
gen_lineitem()
gen_nation()
gen_region()