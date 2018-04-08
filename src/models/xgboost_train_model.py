# -*- coding: utf-8 -*-
import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import xgboost as xgb
import time


def main():
    logger = logging.getLogger(__name__)

    dtypes = {'ip': np.uint32, 'app': np.uint16, 'device_os':np.uint32, 'channel': np.uint8,
              'click_dayofyear': np.uint8, 'click_hour': np.uint8, 'click_minute': np.uint8,
              'is_attributed': np.uint8}
    start_time = time.time()
    X_train_df = pd.read_csv(project_dir + '/data/processed/train_sample.csv', sep=',', dtype=dtypes)
    print("Loading csv took: %s seconds" % (time.time() - start_time))
	
    logger.info("Unique 'ip' count: {0}".format(str(X_train_df.ip.unique().size)))
    logger.info("Unique 'app' count: {0}".format(str(X_train_df.app.unique().size)))
    logger.info("Unique 'channel' count: {0}".format(str(X_train_df.channel.unique().size)))
    logger.info("Total downloads count in training set: {0}".format(str(X_train_df[X_train_df.is_attributed == 1].size)))

    Y_train = X_train_df['is_attributed']
    # drop is_attributed from training dataset
    X_train_df.drop(['is_attributed'], axis=1, inplace=True)
	
	
    start_time = time.time()
    # Fit the model
    model = xgb.XGBRegressor(max_depth=3, n_estimators=100, learning_rate=0.1, silent=False, n_jobs=2)
    model.fit(X_train_df, Y_train)
    print("Model training took: %s seconds" % (time.time() - start_time))

    # save the model to disk
    filename = project_dir + '/models/xgb_regression_model.sav'
    joblib.dump(model, open(filename, 'wb'))

    # 10-fold cross-validation with logistic regression
    # print(cross_val_score(model, X_train_df, y, cv=10, scoring='roc_auc'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    main()
