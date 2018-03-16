import os
import datetime
import pandas as pd
import time


class Session(object):
    def __init__(self, name='', label=''):
        if not label:
            # get a default label
            label = 'unnamed_session'
        pre = name + '_' if name != '' else name
        dir_name = pre + datetime.datetime.now().strftime('%Y%m%dT%H%M%S') + '_' + label.lower().replace(' ', '_')
        self.dir = os.path.join(os.environ['PROJECT_ROOT'], 'output', dir_name)
        os.makedirs(self.dir)

    def save_excel(self, df, filename, tab='excel_save'):
        # force using openpyxl
        filepath = os.path.join(self.dir, filename + '.xlsx')
        ew = pd.ExcelWriter(filepath)
        df.to_excel(ew, tab)
        ew.save()
        tracelog("Excel file written at " + filepath)

    def open_excel(self, filename):
        filepath = os.path.join(self.dir, filename + '.xlsx')
        self.ew = pd.ExcelWriter(filepath)

    def add_tab(self, df, tab):
        df.to_excel(self.ew, tab)

    def close_excel(self):
        self.ew.save()
        tracelog("Excel file written")

    def get_filename(self, filename, ext=None):
        if ext is not None and filename[-4:]!=ext:
            filename = filename + '.' + ext
        return os.path.join(self.dir, filename)


def get_secrets():
    return os.path.join(os.environ['PROJECT_ROOT'], 'config', 'secrets.json')


def tracelog(msg, kind='P'):
    prefix = '[' + kind + '] '
    timelog = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    print(prefix + timelog + ': ' + msg)


def start_stop(func):
    def func_wrapper(*args, **kargs):
        tracelog("{} START".format(func.__name__))
        t0 = time.time()
        out = func(*args, **kargs)
        tracelog("{} STOP (took {} sec)".format(func.__name__, time.time()-t0))
        return out
    return func_wrapper


# from https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
# https://github.com/facebook/prophet/issues/223
# Define a context manager to suppress stdout and stderr.
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull,os.O_RDWR) for _ in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


