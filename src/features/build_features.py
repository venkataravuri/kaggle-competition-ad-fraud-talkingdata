# -*- coding: utf-8 -*-
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import gc


# Credits: https://www.kaggle.com/nanomathias/feature-engineering-importance-testing
def generateAggregateFeatures(train_df, aggregateFeatures):
    logger = logging.getLogger("generateAggregateFeatures")
    for spec in aggregateFeatures:
        logger.info("Generating aggregate feature {} group by {}, \
         and aggregating {} with {}".format(spec['name'], \
                                            spec['groupBy'], spec['select'], spec['agg']))
        gp = train_df[spec['groupBy'] + [spec['select']]] \
            .groupby(by=spec['groupBy'])[spec['select']] \
            .agg(spec['agg']) \
            .reset_index() \
            .rename(index=str, columns={spec['select']: spec['name']})
        train_df = train_df.merge(gp, on=spec['groupBy'], how='left')
        del gp
        gc.collect()

    return train_df


def generateNextClickFeatures(train_df, nextClickAggregateFeatures):
    logger = logging.getLogger("generateNextClickFeatures")
    for spec in nextClickAggregateFeatures:
        feature_name = '{}-next-click'.format('-'.join(spec['groupBy']))
        logger.info("Generating feature '{0}'".format(feature_name))
        train_df[feature_name] = train_df[spec['groupBy'] + ['click_time']].groupby(['ip']).click_time.transform(
            lambda x: x.diff().shift(-1)).dt.seconds
    return train_df


def extractTimeInformation(X_train_df):
    X_train_df['day'] = X_train_df.click_time.dt.day.astype('uint8')
    X_train_df['hour'] = X_train_df.click_time.dt.hour.astype('uint8')
    X_train_df['minute'] = X_train_df.click_time.dt.minute.astype('uint8')
    X_train_df['second'] = X_train_df.click_time.dt.second.astype('uint8')


def main():
    nrows = None
    logger = logging.getLogger(__name__)

    dtypes = {'ip': np.uint32, 'app': np.uint16, 'device': np.uint8, 'os': np.uint8, 'channel': np.uint8,
              'is_attributed': np.uint8}
    X_train_df = pd.read_csv(project_dir + '/data/raw/train.csv', sep=',', nrows=nrows, dtype=dtypes,
                             parse_dates=['click_time', 'attributed_time'])
    logger.info("Total records: {0}".format(str(X_train_df.size)))
    logger.info("Unique 'ip' count: {0}".format(str(X_train_df.ip.unique().size)))
    logger.info("Unique 'app' count: {0}".format(str(X_train_df.app.unique().size)))
    logger.info("Unique 'channel' count: {0}".format(str(X_train_df.channel.unique().size)))
    logger.info("Total downloads count in training set: {0}" \
                .format(str(X_train_df[X_train_df.is_attributed == 1].size)))

    extractTimeInformation(X_train_df)

    aggregateFeatures = [
        # Number of clickes for ip-app
        {'name': 'ip-app-count', 'groupBy': ['ip', 'app'], 'select': 'channel', 'agg': 'count'},
        # Number of clicks for each ip-app-os
        {'name': 'ip-app-os-count', 'groupBy': ['ip', 'app', 'os'], 'select': 'channel', 'agg': 'count'},
        # Number of clicks for ip-day-hour
        {'name': 'ip-day-hour-count', 'groupBy': ['ip', 'day', 'hour'], 'select': 'channel', 'agg': 'count'},
        # Number of clicks for ip-app-day-hour
        {'name': 'ip-app-day-hour-count', 'groupBy': ['ip', 'app', 'day', 'hour'], 'select': 'channel', 'agg': 'count'},
        # Clicks variance in day, for ip-app-channel
        {'name': 'ip-app-channel-var', 'groupBy': ['ip', 'app', 'channel'], 'select': 'day', 'agg': 'var'},
        # Clicks variance in hour, for ip-app-os
        {'name': 'ip-app-os-var', 'groupBy': ['ip', 'app', 'os'], 'select': 'hour', 'agg': 'var'},
        # Clicks variance in hour, for ip-day-channel
        {'name': 'ip-day-channel-var', 'groupBy': ['ip', 'day', 'channel'], 'select': 'hour', 'agg': 'var'},
        # Mean clicks in an hour, for ip-app-channel
        {'name': 'ip-app-channel-mean', 'groupBy': ['ip', 'app', 'channel'], 'select': 'hour', 'agg': 'mean'},
        # How popular is the app in channel?
        {'name': 'app-popularity', 'groupBy': ['app'], 'select': 'channel', 'agg': 'count'},
        # How popular is the channel in app?
        {'name': 'channel-popularity', 'groupBy': ['channel'], 'select': 'app', 'agg': 'count'},
        # Average clicks on app by distinct users; is it an app they return to?
        {'name': 'avg-clicks-on-app', 'groupBy': ['app'], 'select': 'ip',
         'agg': lambda x: float(len(x)) / len(x.unique())}
    ]
    X_train_df = generateAggregateFeatures(X_train_df, aggregateFeatures)

    nextClickAggregateFeatures = [
        {'groupBy': ['ip']},
        {'groupBy': ['ip', 'app']},
        {'groupBy': ['ip', 'channel']},
        {'groupBy': ['ip', 'os']}
    ]
    X_train_df = generateNextClickFeatures(X_train_df, nextClickAggregateFeatures)

    columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', \
               'is_attributed', 'day', 'hour', 'minute', 'second', \
               'ip-app-count', 'ip-app-os-count', 'ip-day-hour-count', 'ip-app-day-hour-count', 'ip-app-channel-var', \
               'ip-app-os-var', 'ip-day-channel-var', 'ip-app-channel-mean', 'app-popularity', 'channel-popularity',
               'avg-clicks-on-app', 'ip-next-click', 'ip-app-next-click', 'ip-channel-next-click', 'ip-os-next-click']

    X_train_df.to_csv(project_dir + '/data/processed/train' + datetime.now().strftime("-%Y-%m-%d-%H-%M-%S") + '.csv',
                      sep=',', columns=columns, index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    main()
