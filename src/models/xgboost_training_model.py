# -*- coding: utf-8 -*-
import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import xgboost as xgb
import time
from sklearn.model_selection import StratifiedKFold


def main():
    logger = logging.getLogger(__name__)

    dtypes = {'ip': np.uint32, 'app': np.uint16, 'device_os': np.uint32, 'channel': np.uint8,
              'click_dayofyear': np.uint8, 'click_hour': np.uint8, 'click_minute': np.uint8,
              'is_attributed': np.uint8}
    start_time = time.time()
    X_train_df = pd.read_csv(project_dir + '/data/processed/train.csv', sep=',', dtype=dtypes)
    logger.info("Loading csv took: %s seconds" % (time.time() - start_time))

    logger.info("Unique 'ip' count: {0}".format(str(X_train_df.ip.unique().size)))
    logger.info("Unique 'app' count: {0}".format(str(X_train_df.app.unique().size)))
    logger.info("Unique 'channel' count: {0}".format(str(X_train_df.channel.unique().size)))
    logger.info(
        "Total downloads count in training set: {0}".format(str(X_train_df[X_train_df.is_attributed == 1].size)))

    kfold = 20
    skf = StratifiedKFold(n_splits=kfold, random_state=42)

    # More parameters has to be tuned. Good luck :)
    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'silent': True,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method' : 'hist',
        'booster': 'gbtree',
        'n_jobs': 4,
        'max_delta_step': 0,
        'colsample_bytree': 1,
        'subsample': 1,
        'min_chil_weight': 1,
        'gamma': 0
    }

    X = X_train_df.drop(['is_attributed'], axis=1).values
    y = X_train_df['is_attributed'].values

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        logger.info('[Fold %d/%d]' % (i + 1, kfold))
        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]
        # Convert our data into XGBoost format
        d_train = xgb.DMatrix(X_train, y_train)
        d_valid = xgb.DMatrix(X_valid, y_valid)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        # Train the model! We pass in a max of 100 rounds (with early stopping after 20)
        # and the custom metric (maximize=True tells xgb that higher metric is better)
        mdl = xgb.train(params, d_train, 10, evals=watchlist, early_stopping_rounds=5, verbose_eval=True)

        logger.info('[Fold %d/%d Prediciton:]' % (i + 1, kfold))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    main()
