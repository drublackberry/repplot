import unittest
import reportlib as rep
import numpy as np
import pandas as pd
from wgutils import tracelog
import time


def check_type(func):
    def func_wrapper(*args, **kargs):
        tracelog("{} START".format(func.__name__))
        t0 = time.time()
        out = func(*args, **kargs)
        tracelog("Return type is {}".format(type(out)))
        tracelog("{} STOP (took {} sec)".format(func.__name__, time.time()-t0))
        return out
    return func_wrapper


df = rep.generate_dummy_data()


class TestUtils(unittest.TestCase):

    @check_type
    def test_plot_cat(self):
        p = rep.plot_cat_bokeh(df, width=900, title='Spam plot in Bokeh')
        self.assertTrue(1)
        return p

    @check_type
    def test_plot_hist(self):
        p = rep.plot_hist_bokeh(df, title='Histogram of SPAM plot in Bokeh')
        self.assertTrue(1)
        return p

    @check_type
    def test_plot_acf(self):
        p = rep.plot_acf_bokeh(df)
        self.assertTrue(1)
        return p

    @check_type
    def test_generate_error_report(self):
         p =  rep.generate_error_report(df, title='Bokeh error report', save_to=None)
         self.assertTrue(1)
         return p

    @check_type
    def test_generate_delta_report(self):
        p = rep.generate_delta_report(df, ['bacon', 'ham'], 'spam', save_to=None)
        self.assertTrue(1)
        return p

    @check_type
    def test_scatter_plot(self):
        p = rep.plot_scatter_bokeh(df, x='spam', y='bacon', c='beans', s='ham', title='Bokeh cluster')
        self.assertTrue(1)
        return p

    @check_type
    def test_boxplot(self):
        p = rep.plot_boxplot_bokeh(df)
        self.assertTrue(1)
        return p


if __name__ == '__main__':
    unittest.main()
