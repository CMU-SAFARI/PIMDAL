import pyarrow
import pyarrow.compute as pc
import numpy as np
import time

repetitions = 10

def query(inner, outer):
    joined_table = inner.join(outer, join_type="inner",
                              keys="key")

if __name__ == "__main__":
    inner = pyarrow.Table.from_arrays(
        [np.arange(1, 512*1024*1024, dtype=np.uint32)],
        ["key"]
    )

    outer = pyarrow.Table.from_arrays(
        [np.random.randint(1, 512*1024*1024, 512*1024*1024, dtype=np.uint32)],
        ["key"]
    )

    start = time.perf_counter()

    for i in range(repetitions):
        query(inner, outer)

    end = time.perf_counter()

    print("Join time CPU:", (end-start)/repetitions)