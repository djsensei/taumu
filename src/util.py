__author__ = 'daniel'

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

TRAINFILE = '../rawfiles/training.csv'
TESTFILE = '../rawfiles/test.csv'

NOT_IN_TEST_COLS = ['mass', 'production', 'min_ANNmuon']

# features whose 1d histograms indicate a visual class separation
SOLO_SIG_FEATURES = ['CDF1', 'CDF2', 'CDF3', 'dira', 'DOCAone', 'IP', 'IPSig',
                     'ISO_SumBDT', 'iso', 'isolationa', 'isolationb', 'isolationc',
                     'isolationd', 'isolatione', 'isolationf', 'p0_eta', 'p0_IsoBDT',
                     'p0_track_Chi2Dof', 'p1_eta', 'p1_IsoBDT', 'p1_track_Chi2Dof',
                     'p2_eta', 'p2_IsoBDT', 'p2_track_Chi2Dof', 'pt', 'SPDhits',
                     'VertexChi2']
SOLO_NONINF_FEATURES = ['DOCAtwo', 'DOCAthree', 'FlightDistance', 'FlightDistanceError',
                        'IP_p0p2', 'IP_p1p2', 'LifeTime', 'p0_p', 'p0_IP', 'p0_IPSig',
                        'p0_pt', 'p1_p', 'p1_IP', 'p1_IPSig', 'p1_pt', 'p2_p', 'p2_IP',
                        'p2_IPSig', 'p2_pt']

"""
Data Loading Functions
"""

def training_df():
    return pd.read_csv(TRAINFILE).drop(NOT_IN_TEST_COLS + ['id'], axis=1).astype(float)

def test_df():
    return pd.read_csv(TESTFILE).astype(float)

def split_labeled_df(df):
    pos = df[df['signal'] == 1]
    neg = df[df['signal'] == 0]
    return pos.drop('signal', axis=1), neg.drop('signal', axis=1)

"""
Visual EDA Functions
"""

# Plots a 1-d histogram+KDE of the given field/feature, showing positive class in blue and negative class in red
def seahist(pdf, ndf, field, save=False):
    f, ax = plt.subplots(figsize=(6, 6))
    sns.distplot(pdf[field].values, bins=50, color='blue', ax=ax)
    sns.distplot(ndf[field].values, bins=50, color='red', ax=ax)
    if save:
        f.savefig('../img/eda/1d_' + field + '.png')

# builds and saves to file all 1d feature histograms. pass in training df!
def prep_all_1d_hists(df):
    os.makedirs('../img/eda/')
    posdf, negdf = split_labeled_df(df)
    for c in df.columns:
        seahist(posdf, negdf, c, True)


"""
Data Prep for ML Functions
"""
def informative_Xtrain(df, scaled=True):
    y = df['signal'].values
    X = df[SOLO_SIG_FEATURES].values
    if scaled:
        ss = StandardScaler()
        ss.fit(X)
        Xs = ss.transform(X)
        return Xs, y, ss
    return X, y, None


def informative_Xtest(df, ss=None):
    X = df[SOLO_SIG_FEATURES].values
    ids = df['id'].values
    if ss is not None:
        Xt = ss.transform(X)
        return Xt, ids
    return X, ids


# Given a model, the training and test dfs, and a name for the version,
#   trains the model, makes predictions, and writes them to file.
def basic_ml_pipeline(model, traindf, testdf, version_name):
    X, y, ss = informative_Xtrain(traindf, True)
    Xtest, testids = informative_Xtest(testdf, ss)
    model.fit(X, y)
    preds = model.predict_proba(Xtest)[:, 1]
    ids_and_preds = zip(testids, preds)
    prep_submission(ids_and_preds, version_name)


"""
Submission Functions
"""
def prep_submission(ids_and_preds, version_name):
    with open("../submissions/" + version_name + ".csv", 'w') as wf:
        wf.write("id,prediction\n")
        for i, p in ids_and_preds:
            wf.write(str(int(i)) + ',' + str(p) + '\n')
