import reportlib as rep
import numpy as np
import time
import pandas as pd
from wgutils import tracelog


def check_type(func):
    def func_wrapper(*args, **kargs):
        tracelog("{} START".format(func.__name__))
        t0 = time.time()
        out = func(*args, **kargs)
        tracelog("Return type is {}".format(type(out)))
        tracelog("{} STOP (took {} sec)".format(func.__name__, time.time()-t0))
        return out
    return func_wrapper


if __name__ == '__main__':
    df = rep.generate_dummy_data()

    @check_type
    def test_plot_cat():
        return rep.plot_cat_bokeh(df, width=900, title='Spam plot in Bokeh')

    @check_type
    def test_plot_hist():
        return rep.plot_hist_bokeh(df, title='Histogram of SPAM plot in Bokeh')

    @check_type
    def test_plot_acf():
        return rep.plot_acf_bokeh(df)

    @check_type
    def test_generate_error_report():
        return rep.generate_error_report(df, title='Bokeh error report', save_to=None)

    @check_type
    def test_generate_delta_report():
        return rep.generate_delta_report(df, ['bacon', 'ham'], 'spam', save_to=None)

    @check_type
    def test_scatter_plot():
        return rep.plot_scatter_bokeh(df, x='spam', y='bacon', c='beans', s='ham', title='Bokeh cluster')

    @check_type
    def test_boxplot():
        return rep.plot_boxplot_bokeh(df)

    p = test_plot_cat()
    p = test_plot_hist()
    p = test_plot_acf()
    p = test_generate_delta_report()
    p = test_generate_error_report()
    p = test_scatter_plot()
    p = test_boxplot()









