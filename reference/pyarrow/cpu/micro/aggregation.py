import pyarrow
import pyarrow.compute as pc
import numpy as np
import time

repetitions = 10

def query(data):
    res = data.group_by("key").aggregate([("val", "sum")])

if __name__ == "__main__":
    data = pyarrow.Table.from_arrays(
        [np.random.randint(1, 50, 1000000000, dtype=np.int32),
        np.random.randint(1, 50, 1000000000, dtype=np.int32)],
        ["key", "val"]
    )

    start = time.perf_counter()

    for i in range(repetitions):
        query(data)

    end = time.perf_counter()

    print("Aggregation time CPU:", (end-start)/repetitions)