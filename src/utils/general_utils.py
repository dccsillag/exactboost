import os
import re
import time
import warnings
from datetime import datetime
from functools import partial

import numpy as np

CANONICAL_DATE_PATTERN = "%Y%m%d"


def string_to_dt(date_str):
    """Converts date in string format to datetime format.

    Args:
        date_str (str): Date in string format.

    Returns:
        datetime.date: Date in datetime format.

    Raises:
        ValueError: if date is in unrecognized pattern.
    """
    date_patterns = ["%Y%m%d", "%Y-%m-%d"]

    for pattern in date_patterns:
        try:
            return datetime.strptime(date_str, pattern)
        except ValueError:
            pass
    raise ValueError(f"Date pattern in string {date_str} is not recognized.")


def float_to_dt(date_float):
    """Converts date in float format to datetime format.

    Args:
        date_float (float): Date in float format.

    Returns:
        datetime.date: Date in datetime format.

    Raises:
        ValueError: if date is in unrecognized pattern.
    """
    date_str = str(date_float)
    date_str = date_str.split(".")[0]
    return string_to_dt(date_str)


def float_to_dt_or_nan(date_float):
    """Converts date in float format to datetime format (or np.nan if conversion fails).

    Args:
        date_float (float): Date in float format.

    Returns:
        datetime.date or np.nan: Date in datetime format or np.nan if conversion fails.
    """
    date_str = str(date_float)
    date_str = date_str.split(".")[0]
    try:
        return string_to_dt(date_str)
    except ValueError:
        return np.nan


def dt_to_string(date_dt):
    """Converts date in datetime format to string format.

    Args:
        date_dt (datetime.date): Date in datetime format.

    Returns:
        str: Date in string format.
    """
    return datetime.strftime(date_dt, CANONICAL_DATE_PATTERN)


def camel_case_to_snake_case(s):
    """Convert string in CamelCase to snake_case.

    Args:
        s (str): String in CamelCase.

    Returns:
        str: String in snake_case.
    """
    _underscorer1 = re.compile(r"(.)([A-Z][a-z]+)")
    _underscorer2 = re.compile("([a-z0-9])([A-Z])")
    subbed = _underscorer1.sub(r"\1_\2", s)
    return _underscorer2.sub(r"\1_\2", subbed).lower()


def save_pandas_to_file(df_to_file_function, out_filepath, verbose):
    """Saves Pandas DataFrame to file.

    Args:
        df_to_file_function (method): Function to save Pandas DataFrame.
        out_filename (str): Output file path.
    """
    # Create output folder if it doesn't exist
    out_folder = os.path.dirname(out_filepath)
    if out_folder != "":
        os.umask(0)
        os.makedirs(out_folder, exist_ok=True, mode=0o777)
    # Save file using provided function
    df_to_file_function(out_filepath)
    # Set correct permissions
    if verbose:
        print("-saved " + out_filepath)


def save_pandas_to_hdf(df, out_filepath, key="data", format="fixed", verbose=True):
    """Saves Pandas DataFrame as hdf file.

    Args:
        df (DataFrame): Pandas DataFrame to store.
        out_filepath (str): Output file path.
        key (str): identifier for the group in the store (always use "data").
        format (str): either 'fixed' (fast writing/reading; not-appendable)
            or 'table' (may perform worse but allow more flexible operations
            like searching/selecting subsets of the data).
    """
    save_pandas_to_file(
        partial(df.to_hdf, key=key, format=format, mode="w"), out_filepath, verbose
    )


def save_pandas_to_feather(df, out_filepath, verbose=True):
    """Saves Pandas DataFrame as feather file.

    Args:
        df (DataFrame): Pandas DataFrame to store.
        out_filepath (str): Output file path.
    """
    save_pandas_to_file(df.to_feather, out_filepath, verbose)


def save_pandas_to_csv(df, out_filepath, index=False, header=True, verbose=True):
    """Save Pandas DataFrame as csv file.

    Args:
        df (DataFrame): Pandas DataFrame to store.
        out_filepath (str): Output file path.
        index (bool): whether to write row names (index).
        header (bool): whether to write out the column names.
    """
    save_pandas_to_file(
        partial(df.to_csv, index=index, header=header), out_filepath, verbose
    )


def get_folder(folder_path, verbose=True):
    """Creates folder, if it doesn't exist, and returns folder path.

    Args:
        folder_path (str): Folder path, either existing or to be created.

    Returns:
        str: folder path.
    """
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        if verbose:
            print(f"-created directory {folder_path}")
    return folder_path


class _TicToc(object):
    """
    Author: Hector Sanchez
    Date: 2018-07-26
    Description: Class that allows you to do 'tic toc' to your code.

    This class was based on https://github.com/hector-sab/ttictoc, which is
    distributed under the MIT license. It prints time information between
    successive tic() and toc() calls.

    Example:

        from src.utils.general_utils import tic,toc

        tic()
        tic()
        toc()
        toc()
    """

    def __init__(self, name="", method="time", nested=False, print_toc=True):
        """
        Args:
            name (str): Just informative, not needed
            method (int|str|ftn|clss): Still trying to understand the default
                options. 'time' uses the 'real wold' clock, while the other
                two use the cpu clock. To use your own method,
                do it through this argument

                Valid int values:
                    0: time.time | 1: time.perf_counter | 2: time.proces_time
                    3: time.time_ns | 4: time.perf_counter_ns
                    5: time.proces_time_ns

                Valid str values:
                  'time': time.time | 'perf_counter': time.perf_counter
                  'process_time': time.proces_time | 'time_ns': time.time_ns
                  'perf_counter_ns': time.perf_counter_ns
                  'proces_time_ns': time.proces_time_ns

                Others:
                  Whatever you want to use as time.time
            nested (bool): Allows to do tic toc with nested with a
                single object. If True, you can put several tics using the
                same object, and each toc will correspond to the respective tic.
                If False, it will only register one single tic, and
                return the respective elapsed time of the future tocs.
            print_toc (bool): Indicates if the toc method will print
                the elapsed time or not.
        """
        self.name = name
        self.nested = nested
        self.tstart = None
        if self.nested:
            self.set_nested(True)

        self._print_toc = print_toc

        self._int2strl = [
            "time",
            "perf_counter",
            "process_time",
            "time_ns",
            "perf_counter_ns",
            "process_time_ns",
        ]
        self._str2fn = {
            "time": [time.time, "s"],
            "perf_counter": [time.perf_counter, "s"],
            "process_time": [time.process_time, "s"],
            "time_ns": [time.time_ns, "ns"],
            "perf_counter_ns": [time.perf_counter_ns, "ns"],
            "process_time_ns": [time.process_time_ns, "ns"],
        }

        if type(method) is not int and type(method) is not str:
            self._get_time = method

        if type(method) is int and method < len(self._int2strl):
            method = self._int2strl[method]
        elif type(method) is int and method > len(self._int2strl):
            self._warning_value(method)
            method = "time"

        if type(method) is str and method in self._str2fn:
            self._get_time = self._str2fn[method][0]
            self._measure = self._str2fn[method][1]
        elif type(method) is str and method not in self._str2fn:
            self._warning_value(method)
            self._get_time = self._str2fn["time"][0]
            self._measure = self._str2fn["time"][1]

    def __warning_value(self, item):
        msg = f"Value '{item}' is not a valid option. Using 'time' instead."
        warnings.warn(msg, Warning)

    def __enter__(self):
        if self.nested:
            self.tstart.append(self._get_time())
        else:
            self.tstart = self._get_time()

    def __exit__(self, type, value, traceback):
        self.tend = self._get_time()
        if self.nested:
            self.elapsed = self.tend - self.tstart.pop()
        else:
            self.elapsed = self.tend - self.tstart

        self._print_elapsed()

    def _print_elapsed(self):
        """
        Prints the elapsed time
        """
        if self.name != "":
            name = "[{}] ".format(self.name)
        else:
            name = self.name
        print(
            "-{0}elapsed time: {1:.3g} ({2})".format(name, self.elapsed, self._measure)
        )

    def tic(self):
        """
        Defines the start of the timing.
        """
        if self.nested:
            self.tstart.append(self._get_time())
        else:
            self.tstart = self._get_time()

    def toc(self, print_elapsed=None):
        """
        Defines the end of the timing.
        """
        self.tend = self._get_time()
        if self.nested:
            if len(self.tstart) > 0:
                self.elapsed = self.tend - self.tstart.pop()
            else:
                self.elapsed = None
        else:
            if self.tstart:
                self.elapsed = self.tend - self.tstart
            else:
                self.elapsed = None

        if print_elapsed is None:
            if self._print_toc:
                self._print_elapsed()
        else:
            if print_elapsed:
                self._print_elapsed()

        # return(self.elapsed)

    def set_print_toc(self, set_print):
        """
        Indicate if you want the timed time printed out or not.

        Args:
          set_print (bool): If True, a message with the elapsed time
            will be printed.
        """
        if type(set_print) is bool:
            self._print_toc = set_print
        else:
            warnings.warn(
                "Parameter 'set_print' not boolean. " "Ignoring the command.", Warning,
            )

    def set_nested(self, nested):
        """
        Sets the nested functionality.
        """
        # Assert that the input is a boolean
        if type(nested) is bool:
            # Check if the request is actually changing the
            # behaviour of the nested tictoc
            if nested != self.nested:
                self.nested = nested

                if self.nested:
                    self.tstart = []
                else:
                    self.tstart = None
        else:
            warnings.warn(
                "Parameter 'nested' not boolean. " "Ignoring the command.", Warning,
            )


class TicToc(_TicToc):
    def tic(self, nested=True):
        """
        Defines the start of the timing.
        """
        if nested:
            self.set_nested(True)

        if self.nested:
            self.tstart.append(self._get_time())
        else:
            self.tstart = self._get_time()


__TICTOC_8320947502983745 = TicToc()
tic = __TICTOC_8320947502983745.tic
toc = __TICTOC_8320947502983745.toc


def import_from_path(path, module_name=None):
    """
    Import (and return) the Python script in `path` as a module.
    """

    import importlib.util

    if module_name is None:
        module_name = "_".join(path.split(os.sep)[-3:-1]).split(".")[0]

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module
