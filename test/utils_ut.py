import unittest
import reportlib as rep
import numpy as np
import pandas as pd

df = pd.DataFrame(columns=['bacon', 'spam', 'beans', 'ham'],
                  index=pd.date_range(start='2015-01-01', end='2018-01-01'))
t = np.array(range(len(df)))
for c in df.columns:
    df.loc[:, c] = np.random.random(1) * np.sin(2 * np.pi * t / len(df)) + np.random.random(len(df))


class TestUtils(unittest.TestCase):

    def test_plot_cat_matplotlib(self):
        rep.plot_cat(df, smooth_period=7, plot_std=True, plot_original=False, title='SPAM plot', marker=None)
        rep.plot_cat(df.iloc[-30:], plot_original=False, title='SPAM plot zoomed',
                     marker='*', show_weekday_on='bacon')
        self.assertTrue(True)  # assess it has finished

    def test_plot_fft_matplotlib(self):
        rep.plot_cat_fft(df[['bacon']])
        self.assertTrue(True)  # assess it has finished
