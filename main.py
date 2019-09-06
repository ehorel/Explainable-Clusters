import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, SpectralClustering, AgglomerativeClustering, OPTICS, KMeans
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import np_utils
from sfit import *
import tensorflow as tf


def prepare_data(df, add_intercept=True):
    if add_intercept:
        df.insert(loc=0, column='intercept', value=1)
    msk = np.random.rand(len(df)) < 0.7
    train = df[msk]
    val_test = df[~msk]
    msk = np.random.rand(len(val_test)) < 0.66
    val = val_test[msk]
    test = val_test[~msk]
    print('Number of observations in train set: {0}'.format(train.shape[0]))
    print('Number of observations in validation set: {0}'.format(val.shape[0]))
    print('Number of observations in test set: {0}'.format(test.shape[0]))
    print('Total number of observations: {0}'.format(train.shape[0] + val.shape[0] + test.shape[0]))
    X_train = train.iloc[:, 0:-1]
    Y_train = train.iloc[:, -1]
    X_val = val.iloc[:, 0:-1]
    Y_val = val.iloc[:, -1]
    X_test = test.iloc[:, 0:-1]
    Y_test = test.iloc[:, -1]
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


if __name__ == "__main__":
    np.random.seed(10)
    tf.compat.v1.random.set_random_seed(10)

    print('Read and prepare data:')
    data_path = './data/firms_ratios_full.csv'
    df = pd.read_csv(data_path)
    df.dropna(axis=0, how='any', inplace=True)
    df = df.groupby('permno').last()
    df = df.iloc[:, 3:]
    df = df.drop('DIVYIELD', axis=1)
    print('Total number of observations: {0}'.format(df.shape[0]))
    print('Number of features: {0}'.format(df.shape[1]))
    df_means = df.mean()
    df_stds = df.std()
    df = (df - df_means)/df_stds

    print('Cluster data:')
    clustering = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    labels = clustering.fit_predict(df.values)

    dict_clusters = {}
    for idx, label in enumerate(clustering.labels_):
        if label in dict_clusters:
            dict_clusters[label].append(idx)
        else:
            dict_clusters[label] = [idx]
    for key in dict_clusters:
        print('Cluster {0} has {1} samples'.format(key, len(dict_clusters[key])))

    print('Prepare data for classification:')
    df['label'] = labels
    X_train, Y_train, X_val, Y_val, X_test, Y_test = prepare_data(df, add_intercept=True)
    Y_train_trans = np_utils.to_categorical(Y_train)
    Y_val_trans = np_utils.to_categorical(Y_val)
    Y_test_trans = np_utils.to_categorical(Y_test)

    print('Fit neural network')
    # Fit 2 hidden layers neural network:
    d = X_train.shape[1]
    nhidden1 = 100
    nhidden2 = 50
    nhidden3 = 25
    batch_size = 32
    nr_epochs = 50
    dropout_rate = 0.3
    inputs = Input(shape=(d,))
    hidden1 = Dense(nhidden1, activation='relu')(inputs)
    # hidden1 = BatchNormalization()(hidden1)
    # hidden1 = Dropout(rate=dropout_rate)(hidden1)
    hidden2 = Dense(nhidden2, activation='relu')(hidden1)
    # hidden2 = BatchNormalization()(hidden2)
    # hidden2 = Dropout(rate=dropout_rate)(hidden2)
    hidden3 = Dense(nhidden3, activation='relu')(hidden2)
    # hidden3 = BatchNormalization()(hidden3)
    # hidden3 = Dropout(rate=dropout_rate)(hidden3)
    output = Dense(5, activation='sigmoid')(hidden3)
    early_stop = EarlyStopping(monitor='val_categorical_accuracy',
                               min_delta=0.001,
                               patience=20)
    reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.1, min_delta=0.0001,
                                  patience=5, min_lr=1e-7)
    model = Model(inputs=inputs, outputs=output)
    optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    model.fit(x=X_train,
              y=Y_train_trans,
              batch_size=batch_size,
              epochs=nr_epochs,
              validation_data=(X_val, Y_val_trans),
              callbacks=[early_stop, reduce_lr],
              verbose=1)

    Y_val_predicted = np.argmax(model.predict(X_val), axis=1)
    nn_val_acc = accuracy_score(Y_val, Y_val_predicted)
    nn_val_bal_acc = balanced_accuracy_score(Y_val, Y_val_predicted)

    print('Neural network accuracy on val set: {0} \n'.format(np.round(nn_val_acc, 2)))
    print('Neural network bal acc on val set: {0} \n'.format(np.round(nn_val_bal_acc, 2)))

    Y_test_predicted = np.argmax(model.predict(X_test), axis=1)
    nn_test_acc = accuracy_score(Y_test, Y_test_predicted)
    nn_test_bal_acc = balanced_accuracy_score(Y_test, Y_test_predicted)

    print('Neural network acc on test set: {0} \n'.format(np.round(nn_test_acc, 2)))
    print('Neural network bal acc on test set: {0} \n'.format(np.round(nn_test_bal_acc, 2)))

    print('Run SFIT on trained model')
    # Compute SFIT on neural network model:
    alpha = 0.05
    beta = 1e-6
    results_sfit_lin = sfit_first_order(model=model,
                                        loss=categorical_cross_entropy_loss,
                                        x=X_test.values,
                                        y=Y_test_trans,
                                        alpha=alpha,
                                        beta=beta)

    significant_var = results_sfit_lin[0]
    median = []
    lower = []
    upper = []
    for key in results_sfit_lin[1]:
        median.append(results_sfit_lin[1][key][0])
        lower.append(results_sfit_lin[1][key][1][0])
        upper.append(results_sfit_lin[1][key][1][1])

    df_dict = {}
    df_dict['variable'] = X_train.columns[significant_var]
    df_dict['median'] = median
    df_dict['CI_lower_bound'] = lower
    df_dict['CI_upper_bound'] = upper
    results = pd.DataFrame.from_dict(df_dict)
    results.sort_values(by='median', ascending=False, inplace=True)
    print(results)



