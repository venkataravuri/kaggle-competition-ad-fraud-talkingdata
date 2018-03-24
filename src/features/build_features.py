# -*- coding: utf-8 -*-
import os
import logging
import pandas as pd
import numpy as np

def main():
    logger = logging.getLogger(__name__)

    dtypes = {'ip': np.uint32, 'app': np.uint16, 'device': np.uint8, 'os': np.uint8, 'channel': np.uint8,
              'is_attributed': np.uint8}
    X_train_df = pd.read_csv(project_dir + '/data/raw/train.csv', sep=',', dtype=dtypes,
                             parse_dates=['click_time', 'attributed_time'])

    logger.info("Unique 'ip' count: {0}".format(str(X_train_df.ip.unique().size)))
    logger.info("Unique 'app' count: {0}".format(str(X_train_df.app.unique().size)))
    logger.info("Unique 'device' & 'os' combinations count: {0}".format(str(X_train_df.groupby(['device', 'os']).size().size)))
    logger.info("Unique 'channel' count: {0}".format(str(X_train_df.channel.unique().size)))
    logger.info("Total downloads count in training set: {0}".format(str(X_train_df[X_train_df.is_attributed == 1].size)))

    X_train_df.drop(['attributed_time'], axis=1, inplace=True)

    X_train_df['device_os'] = X_train_df['device'] * 1000 + X_train_df['os']
    X_train_df.drop(['device'], axis=1, inplace=True)
    X_train_df.drop(['os'], axis=1, inplace=True)

    X_train_df['click_dayofyear'] = X_train_df['click_time'].dt.dayofyear
    X_train_df['click_hour'] = X_train_df['click_time'].dt.hour
    X_train_df['click_minute'] = X_train_df['click_time'].dt.minute
    X_train_df.drop(['click_time'], axis=1, inplace=True)

    columns = ['ip', 'app', 'device_os', 'channel', 'click_dayofyear', 'click_hour', 'click_minute', 'is_attributed']

    X_train_df.to_csv(project_dir + '/data/processed/train.csv', sep=',', columns=columns, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    main()
