import pandas as pd
import numpy as np
import os
import logging
import sys


# Basic logging configuration
logging.basicConfig(format='%(asctime)s %(message)s')


class DataLoader():
    """ DataLoader class containing all data for finetuning
            VARIABLES:
                X_dev, y_dev: X & y for development
                X_test, y_test: X & y for testing at the very end
                X_train, y_train: X & y for training model
                X_pool, y_pool: X & y for active learner queries
                X_unseen, y_unseen: X & y for related, but unseen words
    """
    def __init__(self, arr):
        logging.warning('Importing data from %s' % [x for x in arr if x is not None])

        # annotated data
        train_df = pd.read_csv(arr[0], sep='\t')
        self.X_train = np.array(train_df.loc[0:, 'text'])
        self.y_train = np.array(train_df.loc[0:, 'label'])

        dev_df = pd.read_csv(arr[1], sep='\t')
        self.X_dev = np.array(dev_df.loc[0:, 'text'])
        self.y_dev = np.array(dev_df.loc[0:, 'label'])

        test_df = pd.read_csv(arr[2], sep='\t')
        self.X_test = np.array(test_df.loc[0:, 'text'])
        self.y_test = np.array(test_df.loc[0:, 'label'])

        # unannotated data
        al = pd.read_csv(arr[3], sep='\t')
        self.X_pool = np.array(al.loc[0:, 'text'])
        self.y_pool = np.array(al.loc[0:, 'label'])

        # unseen data, if provided
        if arr[-1] is not None:
            un = pd.read_csv(arr[-1], sep='\t')
            self.X_unseen, self.y_unseen = \
                np.array(un.loc[0:, 'text']), np.array(un.loc[0:, 'label'])
        else:
            self.X_unseen, self.y_unseen = (None, None)
