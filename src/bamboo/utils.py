# Copyright (c) 2025, Tom Ouellette
# Licensed under the GNU GPLv3 License

from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm
from typing import Callable, Iterable, Any


class Map:
    """Apply a function to each element in an iterable.

    Parameters
    ----------
    func : Callable
        An arbitrary function valid for the provided iterable.
    iterable : Iterable
        An iterable object.
    silent : bool
        If True, suppress the progress bar.
    desc : None | str
        Optional user-provided progress bar description.

    Returns
    -------
    Any
        `func` output.
    """

    def __new__(
        cls,
        func: Callable,
        iterable: Iterable,
        silent: bool = False,
        desc: None | str = None,
    ) -> Any:
        tqdm._instances.clear()
        return [func(i) for i in tqdm(iterable, disable=silent, desc=desc)]


class ProcessMap:
    """Apply a function to each element in an iterable in parallel.

    Parameters
    ----------
    func : Callable
        An arbitrary function valid for the provided iterable.
    iterable : Iterable
        An iterable object.
    n_jobs : None | int
        Number of parallel jobs.
    silent : bool
        If True, suppress the progress bar.
    desc : None | str
        Optional user-provided progress bar description.
    kwargs : Any
        Additional arguments to joblib.Parallel.

    Returns
    -------
    Any
        `func` output.
    """

    def __new__(
        cls,
        func: Callable,
        iterable: Iterable,
        n_jobs: None | int = None,
        silent: bool = False,
        desc: None | str = None,
        **kwargs,
    ) -> Any:
        tqdm._instances.clear()
        with parallel_backend("loky"):
            return Parallel(n_jobs=n_jobs, prefer="processes", **kwargs)(
                delayed(func)(i) for i in tqdm(iterable, disable=silent, desc=desc)
            )


class ThreadMap:
    """Apply a function to each element in an iterable using multiple threads.

    Parameters
    ----------
    func : Callable
        An arbitrary function valid for the provided iterable.
    iterable : Iterable
        An iterable object.
    n_jobs : None | int
        Number of parallel jobs.
    silent : bool
        If True, suppress the progress bar.
    desc : None | str
        Optional user-provided progress bar description.
    kwargs : Any
        Additional arguments to ThreadPoolExecutor.

    Returns
    -------
    Any
        `func` output.
    """

    def __new__(
        cls,
        func: Callable,
        iterable: Iterable,
        n_jobs: None | int = None,
        silent: bool = False,
        desc: None | str = None,
        **kwargs,
    ) -> Any:
        tqdm._instances.clear()
        with ThreadPoolExecutor(max_workers=n_jobs, **kwargs) as executor:
            return list(
                tqdm(executor.map(func, iterable), total=len(iterable), desc=desc)
            )


def apply_map(
    func: Callable,
    iterable: Iterable,
    n_jobs: None | int = None,
    prefer: None | str = None,
    silent: bool = False,
    desc: None | str = None,
    **kwargs,
) -> Any:
    """Selects and runs the appropriate map given parallelism settings.

    Parameters
    ----------
    func : Callable
        An arbitrary function valid for the provided iterable.
    iterable : Iterable
        An iterable object.
    n_jobs : None | int
        Number of parallel jobs.
    prefer : None | str
        Either None, 'processes, or 'threads'. If n_jobs and prefer are both
        None, then no parallelism will be used (n_jobs == 1). If n_jobs > 1 and
        prefer is None, then 'processes' will be used. If prefer is 'processes'
        and n_jobs is None, all available processes will be used.
    silent : bool
        If True, suppress the progress bar.
    desc : None | str
        Optional user-provided progress bar description.
    kwargs : Any
        Additional arguments to ThreadPoolExecutor.

    Returns
    -------
    Any
        Iteration output.
    """
    if isinstance(n_jobs, int):
        if n_jobs < 1:
            raise ValueError("'n_jobs' must be greater than 0.")
    else:
        if n_jobs is not None:
            raise TypeError("'n_jobs' must be None or an `int`.")

    if prefer not in [None, "threads", "processes"]:
        raise ValueError("'prefer' must be one of: None, 'threads', or 'processes'.")

    if (n_jobs is None and prefer is None) or n_jobs == 1:
        return Map(func, iterable, silent=silent, desc=desc)

    if prefer is None or prefer == "processes":
        return ProcessMap(
            func, iterable, n_jobs=n_jobs, silent=silent, desc=desc, **kwargs
        )

    return ThreadMap(func, iterable, n_jobs=n_jobs, silent=silent, desc=desc)
