import argparse
import csv
import numpy as np
from sklearn.utils import resample, shuffle

LEARNING_RATE = 0.001
IN = 1
OUT = 0
VERBOSE = 0
_LOG_PRINT = lambda *a: None

def load_cifar10(num_class=10):
    import keras
    # load the pre-shuffled train and test data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # rescale [0,255] --> [0,1]
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255
    # one-hot encoding for the labels
    class_size = len(np.unique(y_train))
    y_train = keras.utils.to_categorical(y_train, class_size)
    y_test = keras.utils.to_categorical(y_test, class_size)
    # reduce the class size
    y_train = y_train[:,:num_class]
    y_test = y_test[:,:num_class]

    shuffle(x_train, y_train, random_state=0)
    shuffle(x_test, y_test, random_state=0)
    return (x_train, y_train), (x_test, y_test)

def sample_data(train_data, test_data, training_size, test_size, num_sets):
    (x_train, y_train), (x_test, y_test) = train_data, test_data
    new_x_train, new_y_train = [], []
    new_x_test, new_y_test = [], []
    for _ in range(num_sets):
        x_temp, y_temp = resample(x_train, y_train, n_samples=training_size, random_state=0)
        new_x_train.append(x_temp)
        new_y_train.append(y_temp)
        x_temp, y_temp = resample(x_test, y_test, n_samples=test_size, random_state=0)
        new_x_test.append(x_temp)
        new_y_test.append(y_temp)
    return (new_x_train, new_y_train), (new_x_test, new_y_test)

def build_cnn_model(num_class=10):
    import keras
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    # build the model
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='tanh', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='tanh'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    if num_class==1:
        model.add(Dense(num_class, activation='sigmoid'))    
    else:
        model.add(Dense(num_class, activation='softmax')) 
    model.summary()
    return model

def get_keras_models(keras_model, num_class, num_models):
    import keras
    models = []
    for i in range(num_models):
        models.append(keras.models.clone_model(keras_model))
        rms = keras.optimizers.RMSprop(lr=LEARNING_RATE, decay=1e-7)
        sgd = keras.optimizers.SGD(lr=LEARNING_RATE, decay=1e-7)
        if num_class == 1 :
            models[i].compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])
        else:
            models[i].compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    return models

def train_keras_models(models, train_data, test_data, epochs):
    (x_train, y_train), (x_test, y_test) = train_data, test_data
    train_accs = []
    test_accs = []
    for i in range(len(models)):
        history = models[i].fit(x_train[i], y_train[i], epochs=epochs, verbose=VERBOSE, shuffle=True, batch_size=32)
        train_accs.append(history.history['acc'][-1])
        score = models[i].evaluate(x_test[i], y_test[i], verbose=VERBOSE)
        test_accs.append(score)
        _LOG_PRINT('\n', 'Model ', i, ' test accuracy:', score[1])
    return (train_accs, test_accs)

def get_attack_dataset(models, train_data, test_data, num_models, data_size):
    # generate dataset for the attack model
    (x_train, y_train), (x_test, y_test) = train_data, test_data
    num_class = len(y_train[0][0])
    x_data, y_data = [[] for _ in range(num_class)], [[] for _ in range(num_class)]
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

def get_trained_svm_models(train_data, test_data):
    from sklearn import svm
    (x_train, y_train), (x_test, y_test) = train_data, test_data
    num_models = len(y_train)
    models = []
    score_sum = 0
    for i in range(num_models):
        _LOG_PRINT('Training svm model : ', i)
        models.append(svm.SVC(gamma='scale',kernel='linear',verbose=VERBOSE))
        models[i].fit(x_train[i], y_train[i])
        score = models[i].score(x_test[i],y_test[i])
        score_sum = score_sum + score
        _LOG_PRINT('SVM model ', i, 'score : ',score)
    _LOG_PRINT('Total attack score : ', score_sum/num_models)
    return models

def get_score_svm_models(models, test_data):
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    (x_test, y_true) = test_data
    acc_scores = []
    pre_scores = []
    rec_scores = []
    for i in range(len(models)):
        y_pred = models[i].predict(x_test[i])
        # _LOG_PRINT(y_pred)
        acc_scores.append(accuracy_score(y_true[i], y_pred))
        pre_scores.append(precision_score(y_true[i], y_pred))
        rec_scores.append(recall_score(y_true[i], y_pred))
    return (acc_scores, pre_scores, rec_scores)

def main(num_target=1, num_shadow=10, training_size=5000, test_size=1000, epochs=10, num_class=10): 
    def split_pair(x_data, y_data, split_point):
        assert len(x_data) == len(y_data)
        assert len(x_data) > split_point
        sp = split_point
        return ((x_data[:sp],y_data[:sp]), (x_data[sp:],y_data[sp:]))

    (x_train, y_train), (x_test, y_test) = load_cifar10(num_class)

    # split the data for each model
    split_point = training_size*num_target
    train_data = split_pair(x_train, y_train, split_point)
    split_point = test_size*num_target
    test_data = split_pair(x_test, y_test, split_point)

    target_train, target_test = sample_data(train_data[0], test_data[0], training_size, test_size, num_target)
    shadow_train, shadow_test = sample_data(train_data[1], test_data[1], training_size, test_size, num_shadow)

    cnn_model = build_cnn_model(num_class)
    # compile the target model
    target_models = get_keras_models(cnn_model, num_class, num_target)
    train_keras_models(target_models, target_train, target_test, epochs)
    # compile the shadow models
    shadow_models = get_keras_models(cnn_model, num_class, num_shadow)
    train_keras_models(shadow_models, shadow_train, shadow_test, epochs)

    # get train data for the attack model
    attack_train = get_attack_dataset(shadow_models, shadow_train, shadow_test, num_shadow, test_size)
    # get test data for the attack model
    attack_test = get_attack_dataset(target_models, target_train, target_test, num_target, test_size)

    # training the attack model
    attack_model = get_trained_svm_models(attack_train, attack_test)

    # TODO generate the report
    scores = get_score_svm_models(attack_model, attack_test)
    _LOG_PRINT(scores)

    return scores

# Experiments
def size_class_exp(num_shadow=100, epochs=100, result_file='result.csv'):
    result = [['training_size']+[i for i in range(1,11)]]
    for training_size in [2500, 5000, 10000, 15000]:
        accuracy, precision, recall = [training_size], [training_size], [training_size]
        for num_class in range(1,11):
            _LOG_PRINT('ts : ', training_size, 'cl : ', num_class)
            scores = main(1,num_shadow,training_size,2000,epochs,num_class)
            accuracy.append(sum(scores[0])/len(scores[0]))
            precision.append(sum(scores[1])/len(scores[1]))
            recall.append(sum(scores[2])/len(scores[2]))
        result.append(accuracy)
        result.append(precision)
        result.append(recall)
    return result

def shadow_num_exp(training_size=5000, test_size=1000, epochs=10, num_class=10, result_file='result.csv'):
    shadow_sizes = [1,10,50,100]
    result = [shadow_sizes]
    accuracy, precision, recall = [], [], []
    for num_shadow in shadow_sizes:
        scores = main(1,num_shadow,training_size,test_size,epochs,num_class)
        accuracy.append(sum(scores[0])/len(scores[0]))
        precision.append(sum(scores[1])/len(scores[1]))
        recall.append(sum(scores[2])/len(scores[2]))
    result.append(accuracy)
    result.append(precision)
    result.append(recall)
    return result

def overfitting_exp(num_shadow=10,training_size=5000, test_size=1000, num_class=10, result_file='result.csv'):
    epochs_sizes = [10, 50, 100, 200, 500]
    result = [epochs_sizes]
    accuracy, precision, recall = [], [], []
    for epochs in epochs_sizes:
        scores = main(1,num_shadow,training_size,test_size,epochs,num_class)
        accuracy.append(sum(scores[0])/len(scores[0]))
        precision.append(sum(scores[1])/len(scores[1]))
        recall.append(sum(scores[2])/len(scores[2]))
    result.append(accuracy)
    result.append(precision)
    result.append(recall)
    return result

if __name__== '__main__':
    parser = argparse.ArgumentParser(description='Member inference attack experiment for CIFAR10')
    parser.add_argument('-t', '--num_target', type=int, default=1)
    parser.add_argument('-s', '--num_shadow', type=int, default=10)
    parser.add_argument('--training_size', type=int, default=5000)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-c', '--num_class', type=int, default=10)
    parser.add_argument('-v', '--verbose', action='count', help='verbose mode')
    parser.add_argument('-r', '--result_file', default='result.csv', type=str, help='file name for the result')
    parser.add_argument('--size_class_exp', action='count', help='experiment with training size and class size')
    parser.add_argument('--shadow_exp', action='count', help='experiment with shadow size')
    parser.add_argument('--overfit_exp', action='count', help='experiment with overfitting')
    args = parser.parse_args()

    if args.verbose:
        _LOG_PRINT = print
        VERBOSE = 1

    if args.size_class_exp:
        result = size_class_exp(args.num_shadow, args.epochs)
    elif args.shadow_exp:
        result = shadow_num_exp(args.training_size, args.test_size, args.epochs, args.num_class)
    elif args.overfit_exp:
        result = overfitting_exp(args.num_shadow,args.training_size, args.test_size, args.num_class)
    else:
        result = main(args.num_target, args.num_shadow, args.training_size, args.test_size, args.epochs, args.num_class)
    
    with open(args.result_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for r in result:
            writer.writerow(r)
