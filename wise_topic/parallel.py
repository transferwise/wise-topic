from typing import Callable, Iterable
import logging
import concurrent.futures

logger = logging.getLogger(__name__)


def process_batch_parallel(
    function: Callable,
    batched_args: Iterable,
    max_workers: int,
) -> list:
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(function, *args): args for args in batched_args
        }

        for future in concurrent.futures.as_completed(future_to_batch):
            args = future_to_batch[future]
            arg_str = ",".join(map(str, args))
            try:
                logger.info(f"Running a task in parallel {arg_str}")
                data = future.result()
                results.append((data, future_to_batch[future]))

            except Exception as ex:
                logger.error(f"Error in running task '{arg_str}': {ex}")

    return results
