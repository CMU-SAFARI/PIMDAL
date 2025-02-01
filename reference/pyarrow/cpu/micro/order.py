import pyarrow
import pyarrow.compute as pc
import numpy as np
import time

repetitions = 5

def query(data):
    res = data.sort_by([("key", "ascending")])

if __name__ == "__main__":
    data = pyarrow.Table.from_arrays(
        [np.random.randint(1, 0xffffffff, 1000000000, dtype=np.uint32)],
        ["key"]
    )

    start = time.perf_counter()

    for i in range(repetitions):
        query(data)

    end = time.perf_counter()

    print("Ordering time CPU:", (end-start)/repetitions)