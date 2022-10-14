import numpy
from hypergbm import make_experiment
from hypernets.tabular.metrics import calc_score

# from hypernets.core.trial import TrialHistory
# from hypernets.searchers import PlaybackSearcher
from hypergbm.search_space import GeneralSearchSpaceGenerator
from hypernets.searchers import EvolutionSearcher
# from hypernets.experiment.cfg import ExperimentCfg as cfg
# cfg.experiment_discriminator = None
# from hypernets.core.callbacks import SummaryCallback

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn import preprocessing

import logging
import numpy as np
import pandas as pd
# import pickle
import joblib
import os


# normalize numerical columns
def normalize(df, cols_to_norm):
    scaler = preprocessing.StandardScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df[cols_to_norm]), columns=cols_to_norm)
    df = df.drop(cols_to_norm, axis=1)
    df = df.join(df_norm)
    # scaler = MinMaxScaler()
    # scaler.fit(x_train)
    # x_train = pd.DataFrame(data=scaler.transform(x_train),index=x_train.index,columns=x_train.columns)
    return df  # .copy()


def inference(estimator, x, y_true, metrics):
    y_pred = estimator.predict(x)
    scores = calc_score(y_true, y_pred, metrics=metrics)
    return scores


def run_experiment(train_data, eval_data, target, enable_lightgbm, enable_xgb, enable_catboost):
    # define search space
    search_space = GeneralSearchSpaceGenerator(n_estimators=300,
                                               enable_lightgbm=enable_lightgbm,
                                               enable_xgb=enable_xgb,
                                               enable_catboost=enable_catboost
                                               )
    # define search algorithm
    searcher = EvolutionSearcher(search_space,
                                 optimize_direction='min',  # rmse
                                 population_size=50,
                                 sample_size=6,
                                 candidates_size=5)
    # create experiment
    experiment = make_experiment(
        train_data=train_data,
        eval_data=eval_data,
        target=target,
        # search_space=search_space,
        cv=False,
        # num_folds=5,
        max_trials=10,
        early_stopping_time_limit=3600,
        log_level=logging.INFO,
        searcher=searcher,
        random_state=7,
        ensemble_size=1,
        collinearity_detection=False,
        feature_generation=False,
        drift_detection=False,  # test data is needed
        feature_selection=True,
        feature_selection_strategy='quantile',
        feature_selection_quantile=0.3,
        reward_metric='rmse',
        webui=True,
        webui_options={
            'event_file_dir': "./events",  # persist experiment running events log to './events'
            'server_port': 8888,  # http server port
            'exit_web_server_on_finish': False  # exit http server after experiment finished
        }
    )
    # sklearn Pipeline on the return https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    estimator = experiment.run()
    return estimator


def main():
    X_columns = ['vol_last_10', 'vol_last_10_MA', 'vol_last_10_EMA', 'vol_last_10_MACD']
    y_column = 'vol_mov'
    model_dirname = './models'
    os.makedirs(model_dirname, exist_ok=True)

    data_filename = './data.csv'
    if os.path.isfile(data_filename) is False:
        print('CSV is not available, check path')
        exit(-1)

    df = pd.read_csv(data_filename)
    print(df.info(), '\n')

    # cross validation
    n_splits = 5
    splits = TimeSeriesSplit(n_splits=n_splits)
    # last test dataset is not used in cross validation
    last_test_set_len = len(df) // (n_splits + 2) + len(df) % (n_splits + 2)
    experiment_idx = 1
    for train_index, val_index in splits.split(df[:-last_test_set_len]):
        print(f'Experiment N{experiment_idx}')
        experiment_idx += 1
        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]
        # final evaluation dataset is unavailable to the model
        # sam size as validation set and next in sequence (shifted)
        test_index = val_index + len(val_index)
        test_df = df.iloc[test_index]
        print(f'Lengths - train:{len(train_index)} val:{len(val_index)} test:{len(test_index)}')
        print(f'Idxs - train {train_index[0]}:{train_index[-1]} val {val_index[0]}:{val_index[-1]} '
              f'test {test_index[0]}:{test_index[-1]}')
        print('\n')

        # use only X,y columns
        train_df = train_df[X_columns + [y_column]]
        val_df = val_df[X_columns + [y_column]]
        test_df = test_df[X_columns + [y_column]]

        # normalize datasets separately
        train_df = normalize(train_df, X_columns)
        val_df = normalize(val_df, X_columns)
        test_df = normalize(test_df, X_columns)

        # init metrics dict
        models = ['lightgbm', 'xgboost', 'catboost']
        metrics = ['rmse', 'mse', 'mae', 'r2']
        results = {}
        for model in models:
            for metric in metrics:
                results[f'{model}_{metric}'] = []

        # train
        for model in models:
            if model == 'lightgbm':
                estimator = run_experiment(train_df, val_df, target=y_column,
                                           enable_lightgbm=True, enable_xgb=False, enable_catboost=False)
            elif model == 'xgboost':
                estimator = run_experiment(train_df, val_df, target=y_column,
                                           enable_lightgbm=False, enable_xgb=True, enable_catboost=False)
            else:
                estimator = run_experiment(train_df, val_df, target=y_column,
                                           enable_lightgbm=False, enable_xgb=False, enable_catboost=True)

            # save model
            joblib.dump(estimator, os.path.join(model_dirname, f'{model}_exp_{experiment_idx}.pkl'))
            # estimator = joblib.load('pipeline.pkl')

            # inference
            scores = inference(estimator, test_df[X_columns], test_df[y_column], metrics)
            print(f'{model} score:{scores}')
            # append scores to average
            for score, value in scores.items():
                results[f'{model}_{score}'].append(value)
    print('Overall performance:')
    # average model scores over all experiments
    avg_results = {}
    for k in results.keys():
        avg_results[k] = np.mean(results[k])
    # print(avg_results)

    # convert to csv
    avg_results_values = numpy.asarray(list(avg_results.values())).reshape(len(models), len(metrics)).astype(np.float32)
    avg_results_df = pd.DataFrame(data=avg_results_values, index=models, columns=metrics)
    print(avg_results_df)
    avg_results_df.to_csv('./results.csv')


if __name__ == '__main__':
    main()
