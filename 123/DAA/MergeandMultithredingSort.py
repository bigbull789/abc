import random
import sys
from contextlib import contextmanager
from multiprocessing import Manager, Pool
from timeit import default_timer as time

class Timer:
    """Record timing information."""

    def __init__(self, *steps):
        self._time_per_step = dict.fromkeys(steps)

    def __getitem__(self, item):
        return self.time_per_step[item]

    @property
    def time_per_step(self):
        return {
            step: elapsed_time
            for step, elapsed_time in self._time_per_step.items()
            if elapsed_time is not None and elapsed_time > 0
        }

    def start_for(self, step):
        self._time_per_step[step] = -time()

    def stop_for(self, step):
        self._time_per_step[step] += time()


def merge_sort_multiple(results, array):
    """Async parallel merge sort."""
    results.append(merge_sort(array))

def multMerge(results, array_part_left, array_part_right):
    """Merge two sorted lists in parallel."""
    results.append(merge(array_part_left, array_part_right))


def merge_sort(array):
    """Perform merge sort."""
    array_length = len(array)
    if array_length <= 1:
        return array
    middle_index = array_length // 2
    left = array[:middle_index]
    right = array[middle_index:]
    left = merge_sort(left)
    right = merge_sort(right)
    return merge(left, right)

def merge(left, right):
    """Merge two sorted lists."""
    sorted_list = [0] * (len(left) + len(right))
    i = j = k = 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            sorted_list[k] = left[i]
            i += 1
        else:
            sorted_list[k] = right[j]
            j += 1
        k += 1

    while i < len(left):
        sorted_list[k] = left[i]
        i += 1
        k += 1

    while j < len(right):
        sorted_list[k] = right[j]
        j += 1
        k += 1

    return sorted_list


@contextmanager
def process_pool(size):
    """Create a process pool and block until all processes have completed."""
    pool = Pool(size)
    yield pool
    pool.close()
    pool.join()

def para_merge_sort(array, ps_count):
    """Perform parallel merge sort."""
    timer = Timer("sort", "merge", "total")
    timer.start_for("total")
    timer.start_for("sort")

    length = len(array)
    step = int(length / ps_count)  # Divide the list in chunks

    # Instantiate a multiprocessing.Manager obj to store the output
    manager = Manager()
    res = manager.list()

    with process_pool(size=ps_count) as pool:
        for i in range(ps_count):
            # Split array into chunks
            if i < ps_count - 1:
                chunk = array[i * step: (i + 1) * step]
            else:
                # Get remaining elements
                chunk = array[i * step:]
            pool.apply_async(merge_sort_multiple, (res, chunk))

        pool.close()
        pool.join()

    timer.stop_for("sort")

    print("Performing final merge.")
    timer.start_for("merge")

    # Merge sublists in parallel
    while len(res) > 1:
        with process_pool(size=ps_count) as pool:
            pool.apply_async(multMerge, (res, res.pop(0), res.pop(0)))

    timer.stop_for("merge")
    timer.stop_for("total")

    final_sorted_list = res[0]
    return timer, final_sorted_list

def get_command_line_parameters():
    """Get the process count from command line parameters (safe for Jupyter)."""
    total_processes = 1  # default

    if len(sys.argv) > 1:
        try:
            total_processes = int(sys.argv[1])
            if total_processes > 1:
                # Restrict process count to even numbers
                if total_processes % 2 != 0:
                    print("Process count should be an even number.")
                    sys.exit(1)
                print(f"Using {total_processes} cores")
            else:
                total_processes = 1
        except ValueError:
            # Handles non-integer arguments (like '-f' in Jupyter)
            total_processes = 1

    return {"pc": total_processes}


if __name__ == "__main__":
    parameters = get_command_line_parameters()
    pc = parameters["pc"]

    main_timer = Timer("single_core", "list_generation")
    main_timer.start_for("list_generation")

    length = random.randint(3 * 10**6, 4 * 10**6)
    randArr = [random.randint(0, i * 100) for i in range(length)]

    main_timer.stop_for("list_generation")

    print(f"List length: {length}")
    print(f"Random generated in {main_timer['list_generation']:.6f} sec")

    main_timer.start_for("single_core")
    single = merge_sort(randArr)
    main_timer.stop_for("single_core")

    randArr_sorted = randArr[:]  # Create a copy due to mutation
    randArr_sorted.sort()

    print("Verification of sorting algo:", randArr_sorted == single)
    print(f"Single Core elapsed time: {main_timer['single_core']:.6f} sec")

    print("Starting parallel sort.")
    para_timer, para_sorted_list = para_merge_sort(randArr, pc)

    print(f"Final merge duration: {para_timer['merge']:.6f} sec")
    print("Sorted arrays equal:", para_sorted_list == randArr_sorted)
    print(f"{pc}-Core elapsed time: {para_timer['total']:.6f} sec")
