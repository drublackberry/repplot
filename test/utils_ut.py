import unittest
import utils
import numpy as np
import pandas as pd


class TestUtils(unittest.TestCase):

    def test_cluster(self):



        self.assertTrue('cluster' in df_out.index.names)
        self.assertTrue(cluster.time_label in df_out.index.names)

        df_out.loc[:, 'ecpc'] = df_out.revenue / df_out.volume
        df_orig = cluster.transform(df_out.ecpc)

        self.assertEqual((df.xs('Booking.com', axis=0, level='agency').index == df_orig.index).sum(), 1)






