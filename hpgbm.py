from hypergbm import make_experiment
from hypernets.tabular.metrics import calc_score

# from hypernets.core.trial import TrialHistory
# from hypernets.searchers import PlaybackSearcher
from hypergbm.search_space import GeneralSearchSpaceGenerator
from hypernets.searchers import EvolutionSearcher
# from hypernets.experiment.cfg import ExperimentCfg as cfg
# cfg.experiment_discriminator = None
# from hypernets.core.callbacks import SummaryCallback

# from sklearn.model_selection import train_test_split
import logging
import numpy as np
import pandas as pd
import pickle


def main():
    filename = './train.csv'
    # df = pd.read_csv(filename, index_col=0).dropna()
    # print(df)
    # print(df.info())
    # y = df['vol_mov']
    # X = df.drop('vol_mov', axis=1)
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=100, shuffle=False
    # )

    # difine search_space, use lightgbm, xgboost and catboost
    search_space = GeneralSearchSpaceGenerator(n_estimators=300, enable_lightgbm=True, enable_xgb=True,
                                               # enable_catboost=True,
                                               # catboost_init_kwargs={'random_state': 8, 'n_estimators': 200}
                                               )
    # define search_algorithm
    searcher = EvolutionSearcher(search_space, optimize_direction='max', population_size=50, sample_size=6,
                                 candidates_size=5)
    # history = TrialHistory.load_history(search_space, './model.pkl')
    # searcher = PlaybackSearcher(history, top_n=1, optimize_direction='max')

    experiment = make_experiment(
        train_data=filename,
        test_data='./test.csv',
        target='vol_mov',
        cv=True,
        num_folds=5,
        max_trials=1,
        early_stopping_time_limit=3600,
        log_level=logging.INFO,
        searcher=searcher,
        random_state=7,
        ensemble_size=30,
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
    print(estimator)

    # save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(estimator, f)

    # predict
    y_pred = estimator.predict(
        pd.DataFrame([[123354, 123354], [123354, 2432], [335355, 124]], columns=['vol_prev_10', 'vol_next_10']))
    y_true = np.asarray([0.2314, 1.2342, -4.4325])
    score = calc_score(y_true, y_pred, metrics=['rmse', 'mse', 'mae', 'r2'])
    print('evaluate score:', score)


if __name__ == '__main__':
    main()
