import keras
import numpy as np
from sklearn.utils import resample

LEARNING_RATE = 0.001
#EPOCH = 100
EPOCH = 100
DATA_SIZE = 5000
TRAINING_SIZE = 3000
TEST_SIZE = 2000
NUM_TARGET = 1
#NUM_SHADOW = 100
NUM_SHADOW = 20
IN = 1
OUT = 0
VERBOSE = 1

def load_data(path='UCI.adult.data'):
    from keras.utils.data_utils import get_file
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    """Loads the UCI dataset.
    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = get_file(path,
                    origin='http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
    f = pd.read_csv(path, header=None)
    # encode the categorical values
    for col in [1,3,5,6,7,8,9,13,14]:
        le = LabelEncoder()
        f[col] = le.fit_transform(f[col].astype('str'))
    # normalize the values
    x_range = [i for i in range(14)]
    f[x_range] = f[x_range]/f[x_range].max()

    x = f[x_range].values
    y = f[14].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)
    return (x_train, y_train), (x_test, y_test)

def sample_data(train_data,test_data,num_sets):
    (x_train, y_train), (x_test, y_test) = train_data, test_data
    new_x_train, new_y_train = [], []
    new_x_test, new_y_test = [], []
    for i in range(num_sets):
        x_temp, y_temp = resample(x_train, y_train, n_samples=TRAINING_SIZE, random_state=0)
        new_x_train.append(x_temp)
        new_y_train.append(y_temp)
        x_temp, y_temp = resample(x_test, y_test, n_samples=TEST_SIZE, random_state=0)
        new_x_test.append(x_temp)
        new_y_test.append(y_temp)
    return (new_x_train, new_y_train), (new_x_test, new_y_test)

def build_fcnn_model():
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    # build the model
    model = Sequential()
    model.add(Dense(512, input_dim=14, activation='tanh'))
    model.add(Dense(512, activation='tanh'))
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    return model

def get_trained_keras_models(keras_model, train_data, test_data, num_models):
    (x_train, y_train), (x_test, y_test) = train_data, test_data
    models = []
    for i in range(num_models):
        models.append(keras.models.clone_model(keras_model))
        rms = keras.optimizers.RMSprop(lr=LEARNING_RATE, decay=1e-7)
        models[i].compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])
        models[i].fit(x_train[i], y_train[i], batch_size=32, epochs=EPOCH, verbose=VERBOSE, shuffle=True)
        score = models[i].evaluate(x_test[i], y_test[i], verbose=VERBOSE)
        print('\n', 'Model ', i, ' test accuracy:', score[1])
    return models

def get_attack_dataset(models, train_data, test_data, num_models, data_size):
    # generate dataset for the attack model
    (x_train, y_train), (x_test, y_test) = train_data, test_data
    num_classes = 1
    x_data, y_data = [[] for i in range(num_classes)], [[] for i in range(num_classes)]
    for i in range(num_models):
        # IN data
        x_temp, y_temp = resample(x_train[i], y_train[i], n_samples=data_size, random_state=0)
        for j in range(data_size):
            y_idx = np.argmax(y_temp[j])
            x_data[y_idx].append(models[i].predict(x_temp[j:j+1])[0])
            y_data[y_idx].append(IN)
        # OUT data
        x_temp, y_temp = resample(x_test[i], y_test[i], n_samples=data_size, random_state=0)
        for j in range(data_size):
            y_idx = np.argmax(y_temp[j])
            x_data[y_idx].append(models[i].predict(x_temp[j:j+1])[0])
            y_data[y_idx].append(OUT)
    return x_data, y_data

def get_trained_svm_models(train_data, test_data, num_models=1):
    from sklearn import svm
    (x_train, y_train), (x_test, y_test) = train_data, test_data
    models = []
    for i in range(num_models):
        print('Training svm model : ', i)
        models.append(svm.SVC(gamma='scale',kernel='linear',verbose=VERBOSE))
        models[i].fit(x_train[i], y_train[i])
        score = models[i].score(x_test[i],y_test[i])
        print('SVM model ', i, 'score : ',score)
    return models

def main():
    print('Hello World!')
    # load the pre-shuffled train and test data
    (x_train, y_train), (x_test, y_test) = load_data()

    # split the data for each model
    target_train = (x_train[:TRAINING_SIZE*NUM_TARGET],y_train[:TRAINING_SIZE*NUM_TARGET])
    target_test = (x_test[:TEST_SIZE*NUM_TARGET],y_test[:TEST_SIZE*NUM_TARGET])
    target_train_data, target_test_data = sample_data(target_train, target_test, NUM_TARGET)

    shadow_train = (x_train[TRAINING_SIZE*NUM_TARGET:],y_train[TRAINING_SIZE*NUM_TARGET:])
    shadow_test = (x_test[TEST_SIZE*NUM_TARGET:],y_test[TEST_SIZE*NUM_TARGET:])
    shadow_train_data, shadow_test_data = sample_data(shadow_train, shadow_test, NUM_SHADOW)

    cnn_model = build_fcnn_model()
    # compile the target model
    target_models = get_trained_keras_models(cnn_model, target_train_data, target_test_data, NUM_TARGET)
    # compile the shadow models
    shadow_models = get_trained_keras_models(cnn_model, shadow_train_data, shadow_test_data, NUM_SHADOW)

    # get train data for the attack model
    attack_train = get_attack_dataset(shadow_models, shadow_train_data, shadow_test_data, NUM_SHADOW, TEST_SIZE)
    # get test data for the attack model
    attack_test = get_attack_dataset(target_models, target_train_data, target_test_data, NUM_TARGET, TEST_SIZE)

    # training the attack model
    attack_model = get_trained_svm_models(attack_train, attack_test)

    # TODO generate the report


if __name__== '__main__':
    main()