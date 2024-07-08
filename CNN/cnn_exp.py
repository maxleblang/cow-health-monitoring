import os
import os.path
import logging
logging.getLogger('tensorflow').disabled = True

from numpy import mean
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from matplotlib import pyplot as plt
from pandas import merge
import numpy as np
import pywt
import seaborn as sns

from load_data_utils import load_to_df, create_rolling_window_data
from models import CNN, CNN2D, multihead_CNN
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
# Copied from internet to get rid of error
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from config import Config
accel_dir = Config.ACCELEROMETER_DATA_DIR
uwb_dir = Config.UWB_DATA_DIR
labels_dir = Config.BEHAVIOR_LABELS_DATA_DIR
data_bin_dir = Config.NUMPY_DATA_BIN_PATH

gpu_config = ConfigProto()
gpu_config.gpu_options.allow_growth = True
session = InteractiveSession(config=gpu_config)


def build_multihead_model(train_data: tuple, val_data: tuple, test_data: tuple, class_weights: dict = None):
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Unpack data
    trainX_accel, trainX_uwb, trainy = train_data
    valX_accel, valX_uwb, valy = val_data
    testX_accel, testX_uwb, testy = test_data

    verbose, epochs, batch_size = 0, 15, 64
    n_timesteps, n1_features, n2_features, n_outputs = trainX_accel.shape[1], trainX_accel.shape[2], trainX_uwb.shape[2], trainy.shape[1]

    model = multihead_CNN(n_timesteps, n1_features, n2_features, n_outputs)

    # fit network
    model.fit([trainX_accel, trainX_uwb], trainy, epochs=epochs, batch_size=batch_size,
              verbose=verbose, callbacks=[early_stop],
              validation_data=([valX_accel, valX_uwb], valy),
              class_weight=class_weights)
    
    # evaluate model
    _, accuracy, f1_score = model.evaluate(
        [testX_accel, testX_uwb], testy, batch_size=batch_size, verbose=0)
    
    # get_confusion_matrix(model, testX, testy)
    class_acc = get_class_accuracies(model, testX_accel, testX_uwb, testy)
    
    return accuracy, f1_score, class_acc

def get_class_accuracies(model, accel_testX, uwb_testX, testy):
    y_pred = model.predict([accel_testX, uwb_testX])

    # One hot encoding back to single int
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(testy, axis=1)

    conf_matrix = confusion_matrix(y_test_classes, y_pred_classes, labels=range(7))

    num_classes = conf_matrix.shape[0]
    acc_per_class = np.zeros(num_classes)
    
    for i in range(num_classes):
        true_positive = conf_matrix[i, i]
        false_negative = np.sum(conf_matrix[i, :]) - true_positive
        acc_per_class[i] = true_positive / (true_positive + false_negative)
    
    # Print accuracy for each class
    # for i, acc in zip(range(num_classes), acc_per_class):
    #     print(f"Class {i+1}: {acc:.2f}")
    
    return np.nan_to_num(acc_per_class) # Make sure we have a number value for all classes


def get_confusion_matrix(model, testX, testy):
    y_pred = model.predict(testX)

    # One hot encoding back to single int
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(testy, axis=1)

    conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['1', '2', '3', '4','5','6','7'], yticklabels=['1', '2', '3', '4','5','6','7'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('raw_accel_conf_matrix.png')


def run_exp(repeats=3):
    tags = [1,2,3,4,5,6,7,8,9,10]
    # The tag numbers we want to train on
    train_tags = [1,2,3,4,5,6]
    # The tag numbers we want to use as validation
    val_tags = [9,10]
    # Use tuples to indicate combining tags
    # The tag numbers we want to test on
    test_tags = [(7,8)]

    # Pre define the arrays to hold data so the index of the
    # data corresponds to the tag number it came from
    accel_data = [list()] * (max(tags) + 1)
    uwb_data = [list()] * (max(tags) + 1)
    behavior_label = [list()] * (max(tags) + 1)

    # Load in all the training
    print("LOAD DATA")
    for tag in tags:
        # See if the data is already saved
        if os.path.isfile(data_bin_dir):
            print(f"Using data saved in {data_bin_dir}")
            break

        # Build folder path
        accel_data_dir = accel_dir + 'T' + str(tag).zfill(2) + '/'
        uwb_data_dir = uwb_dir + 'T' + str(tag).zfill(2) + '/'

        # Get all accelerometer sensor data files for this folder
        accel_filepaths = os.listdir(accel_data_dir)
        accel_filepaths = [accel_data_dir + file for file in accel_filepaths if file.endswith('0725.csv')]
        accel_filepaths.sort() # Make sure they're in order for processing

        # Get all UWB sensor data files for this folder
        uwb_filepaths = os.listdir(uwb_data_dir)
        uwb_filepaths = [uwb_data_dir + file for file in uwb_filepaths if file.endswith('0725.csv')]
        uwb_filepaths.sort() # Make sure they're in order for processing
        
        # Get groundtruth path
        groundtruth_path = labels_dir + 'C' + str(tag).zfill(2) + '_0725.csv'

        # Load in both sensor data
        accel_input_df, groundtruth_df = load_to_df(accel_filepaths, groundtruth_path, standardized_delta=0.1)
        uwb_input_df, _ = load_to_df(uwb_filepaths, groundtruth_path, standardized_delta=0.1)

        # Convert uwb from cm to m
        uwb_input_df['coord_x_cm'] = uwb_input_df['coord_x_cm']/100
        uwb_input_df['coord_y_cm'] = uwb_input_df['coord_y_cm']/100
        uwb_input_df['coord_z_cm'] = uwb_input_df['coord_z_cm']/100

        # Combine all sensor data together
        # input_df = merge(accel_input_df, uwb_input_df, how='outer', on='timestamp')

        # Create sliding window
        accel_X, y = create_rolling_window_data(accel_input_df, groundtruth_df, window_size=10, stride=5)
        uwb_X, _ = create_rolling_window_data(uwb_input_df, groundtruth_df, window_size=10, stride=5)

        # Add to array
        accel_data[tag] = accel_X
        uwb_data[tag] = uwb_X
        behavior_label[tag] = y

        print(f"Loaded in tag {tag}")


    if os.path.isfile(data_bin_dir):
        # Load in binary
        npzfile = np.load(data_bin_dir, allow_pickle=True)
        accel_data = npzfile['accel']
        uwb_data = npzfile['uwb']
        behavior_label = npzfile['label']
    
    else:
        # Save data in file
        np.savez(data_bin_dir, accel=np.array(accel_data), uwb=np.array(uwb_data), label=np.array(behavior_label))

    # Loop through every test tag to train on all other data and test on this one
    print("\nTESTING")

    overall_class_accuracies = np.zeros(7)
    for test_tuple in test_tags:
        print(f"\nTesting on tag group of {str(test_tuple)}")

        '''
        Prepare TRAIN data
        - All training data gets combined
        '''
        # Combine all training data
        accel_X_train = np.array([])
        uwb_X_train = np.array([])
        y_train = np.array([])

        for train_tag in train_tags:
            # Skip if this is the tag we're testing on
            if train_tag in test_tuple:
                continue
            
            # Add sensor data
            if accel_X_train.shape[0] == 0:
                accel_X_train = np.copy(accel_data[train_tag])
                uwb_X_train = np.copy(uwb_data[train_tag])
            else:
                accel_X_train = np.vstack([accel_X_train, accel_data[train_tag]])
                uwb_X_train = np.vstack([uwb_X_train, uwb_data[train_tag]])
            
            # Add behavior label
            if y_train.shape[0] == 0:
                y_train = np.copy(behavior_label[train_tag])
            else:
                y_train = np.concatenate((y_train, behavior_label[train_tag]))
                
        # One-hot encoding
        y_train = to_categorical(y_train - 1, num_classes = 7)


        '''
        Prepare VALIDATION data
        - All validation data gets combined
        '''
        # Combine all validation data
        accel_X_val = np.array([])
        uwb_X_val = np.array([])
        y_val = np.array([])

        # Make sure we're only validating with one group of tags
        for val_tag in val_tags:
            # Skip if this is the tag we're testing on
            if val_tag in test_tuple:
                continue
            
            # Add sensor data
            if accel_X_val.shape[0] == 0:
                accel_X_val = np.copy(accel_data[val_tag])
                uwb_X_val = np.copy(uwb_data[val_tag])
            else:
                accel_X_val = np.vstack([accel_X_val, accel_data[val_tag]])
                uwb_X_val = np.vstack([uwb_X_val, uwb_data[val_tag]])
            
            # Add behavior label
            if y_val.shape[0] == 0:
                y_val = np.copy(behavior_label[val_tag])
            else:
                y_val = np.concatenate((y_val, behavior_label[val_tag]))
                
        # One-hot encoding
        y_val = to_categorical(y_val - 1, num_classes = 7)


        '''
        Prepare TESTING data
        - All testing data in the TUPLE gets combined
        '''
        # Combine the testing data in the tuple
        accel_X_test = np.array([])
        uwb_X_test = np.array([])
        y_test = np.array([])

        for test_tag in test_tuple:
            # Add sensor data
            if accel_X_test.shape[0] == 0:
                accel_X_test = np.copy(accel_data[test_tag])
                uwb_X_test = np.copy(uwb_data[test_tag])
            else:
                accel_X_test = np.vstack([accel_X_test, accel_data[test_tag]])
                uwb_X_test = np.vstack([uwb_X_test, uwb_data[test_tag]])
            
            # Add behavior label
            if y_test.shape[0] == 0:
                y_test = np.copy(behavior_label[test_tag])
            else:
                y_test = np.concatenate((y_test, behavior_label[test_tag]))
                
        # One-hot encoding
        y_test = to_categorical(y_test - 1, num_classes = 7)

        # BANDAID FIX for if number of samples is different
        train_size = min(uwb_X_train.shape[0],accel_X_train.shape[0])
        val_size = min(uwb_X_val.shape[0],accel_X_val.shape[0])
        test_size = min(uwb_X_test.shape[0],accel_X_test.shape[0])

        accel_X_train = accel_X_train[:train_size]
        uwb_X_train = uwb_X_train[:train_size]
        y_train = y_train[:train_size]

        accel_X_val = accel_X_val[:val_size]
        uwb_X_val = uwb_X_val[:val_size]
        y_val = y_val[:val_size]

        accel_X_test = accel_X_test[:test_size]
        uwb_X_val = uwb_X_val[:val_size]
        y_test = y_test[:test_size]

        # Train/Test split for data
        print("Training data shape: (X accel) (X uwb) (y)")
        print(accel_X_train.shape, uwb_X_train.shape, y_train.shape)
        print("Validation data shape: (X accel) (X uwb) (y)")
        print(accel_X_val.shape, uwb_X_val.shape, y_val.shape)
        print("Testing data shape: (X accel) (X uwb) (y)")
        print(accel_X_test.shape, uwb_X_test.shape, y_test.shape)


        '''
        Run experiment
        '''
        # Class balancing
        true_labels_train = np.argmax(y_train, axis=1)
        class_weights = compute_class_weight('balanced', classes=np.unique(true_labels_train), y=true_labels_train)
        class_weights_dict = {i: weight/7 for i, weight in enumerate(class_weights)}


        accuracies = list()
        f1_scores = list()
        class_accuracies = np.zeros(7)
        print('      F1    \tAcc')
        for r in range(repeats):
            acc, f1_score, class_acc = build_multihead_model((accel_X_train, uwb_X_train, y_train),
                                                             (accel_X_val, uwb_X_val, y_val),
                                                             (accel_X_test, uwb_X_test, y_test),
                                                             class_weights=class_weights_dict)
            acc = acc * 100.0
            f1_score = f1_score * 100.0
            class_acc = class_acc * 100.0
            print('>#%d:  %.3f\t%.3f' % (r+1, acc, f1_score))
            accuracies.append(acc)
            f1_scores.append(f1_score)
            class_accuracies += class_acc
        # model results
        print('      %.3f\t%.3f\n' % (mean(accuracies), mean(f1_scores)))
        # Class results
        class_accuracies = class_accuracies / repeats
        print(class_accuracies)
        overall_class_accuracies += class_accuracies


    for class_i, acc in zip(range(7), overall_class_accuracies):
        print('Class %d: %.3f' % (class_i+1, acc))



def run_wv_exp(repeats=3):
    accel_data_prefix = 'converted_data/'
    uwb_data_prefix = 'location_data/'

    # The tag numbers we want to train on
    # train_tags = [1,2,3,4,5,6,8,9,10]
    train_tags = [3,5,8,9,10]
    # The tag numbers we want to test on
    # test_tags = [1,2,3,4,5,6,7,8,9,10]
    test_tags = [7]

    # Array of all the training data we load in from each tag
    train_inputs = []
    train_groundtruths = []
    # Array of all the testing data we load in from each tag
    test_inputs = []
    test_groundtruths = []

    # Wavelet constants
    waveletname = "mexh"
    scales = range(1,201)


    # Load in all the training
    for tag in train_tags:
        # Build folder path
        accel_data_dir = accel_data_prefix + 'T' + str(tag).zfill(2) + '/'
        uwb_data_dir = uwb_data_prefix + 'T' + str(tag).zfill(2) + '/'
        groundtruth_dir = 'behavior_analysis/individual_behaviors/'

        # Get all accelerometer sensor data files for this folder
        accel_filepaths = os.listdir(accel_data_dir)
        accel_filepaths = [accel_data_dir + file for file in accel_filepaths if file.startswith('sensor_data') and file.endswith('0725.csv')]
        accel_filepaths.sort() # Make sure they're in order for processing

        # Get all UWB sensor data files for this folder
        uwb_filepaths = os.listdir(uwb_data_dir)
        uwb_filepaths = [uwb_data_dir + file for file in uwb_filepaths if file.startswith('uwb_loc') and file.endswith('0725.csv')]
        uwb_filepaths.sort() # Make sure they're in order for processing
        
        # Get groundtruth path
        groundtruth_path = groundtruth_dir + 'C' + str(tag).zfill(2) + '_0725.csv'

        # Load in both sensor data
        accel_input_df, groundtruth_df = load_to_df(accel_filepaths, groundtruth_path)
        uwb_input_df, _ = load_to_df(uwb_filepaths, groundtruth_path)

        # Combine all sensor data together
        input_df = merge(accel_input_df, uwb_input_df, how='outer', on='timestamp')

        print(f"Loaded in tag {tag}")
        # Create sliding window
        X, y = create_rolling_window_data(input_df, groundtruth_df,window_size=20, stride=10)
        print(f"Created Sliding window for tag {tag}")

        channels = X.shape[2]

        # Create wavelets
        wvs = np.ndarray((X.shape[0],200,200,channels))

        for wi in range(len(X)):
            # Transpose into X, Y, and Z channels
            window = X[wi].T
            for si in range(channels):
                signal = window[si]
                coeff, _ = pywt.cwt(signal, scales, waveletname, 1)
                wvs[wi, :, :, si] = coeff
        
        print(f"Created CWT for tag {tag} \n")

        # Add to array
        train_inputs.append(wvs)
        train_groundtruths.append(y)


    # Load in all the testing (just one day)
    for tag in test_tags:
        # Build folder path
        accel_data_dir = accel_data_prefix + 'T' + str(tag).zfill(2) + '/'
        uwb_data_dir = uwb_data_prefix + 'T' + str(tag).zfill(2) + '/'
        groundtruth_dir = 'behavior_analysis/individual_behaviors/'

        # Get all accelerometer sensor data files for this folder
        accel_filepaths = os.listdir(accel_data_dir)
        accel_filepaths = [accel_data_dir + file for file in accel_filepaths if file.startswith('sensor_data') and file.endswith('0725.csv')]
        accel_filepaths.sort() # Make sure they're in order for processing

        # Get all UWB sensor data files for this folder
        uwb_filepaths = os.listdir(uwb_data_dir)
        uwb_filepaths = [uwb_data_dir + file for file in uwb_filepaths if file.startswith('uwb_loc') and file.endswith('0725.csv')]
        uwb_filepaths.sort() # Make sure they're in order for processing
        
        # Get groundtruth path
        groundtruth_path = groundtruth_dir + 'C' + str(tag).zfill(2) + '_0725.csv'

        # Load in both sensor data
        accel_input_df, groundtruth_df = load_to_df(accel_filepaths, groundtruth_path)
        uwb_input_df, _ = load_to_df(uwb_filepaths, groundtruth_path)

        # Combine all sensor data together
        input_df = merge(accel_input_df, uwb_input_df, how='outer', on='timestamp')

        print(f"Loaded in tag {tag}")
        # Create sliding window
        X, y = create_rolling_window_data(input_df, groundtruth_df,window_size=20,stride=10)
        print(f"Created Sliding window for tag {tag}")

        channels = X.shape[2]

        # Create wavelets
        wvs = np.ndarray((X.shape[0],200,200,channels))

        for wi in range(len(X)):
            # Transpose into X, Y, and Z channels
            window = X[wi].T
            for si in range(channels):
                signal = window[si]
                coeff, _ = pywt.cwt(signal, scales, waveletname, 1)
                wvs[wi, :, :, si] = coeff

        print(f"Created CWT for tag {tag} \n")

        # Add to array
        test_inputs.append(wvs)
        test_groundtruths.append(y)

    accuracies = []
    # Loop through every test tag to train on all other data and test on this one
    for test_tag_i in range(len(test_tags)):
        print(f"Testing on tag {test_tags[test_tag_i]}")

        # Prepare training data
        # Combine all training data
        X_train = np.array([])
        y_train = np.array([])

        for i in range(len(train_tags)):
            # Skip if this is the tag we're testing on
            if train_tags[i] == test_tags[test_tag_i]:
                continue

            # X_train += train_inputs[i]
            # y_train += train_groundtruths[i]
            if X_train.shape[0] == 0:
                X_train = np.copy(train_inputs[i])
            else:
                X_train = np.vstack([X_train, train_inputs[i]])
            
            if y_train.shape[0] == 0:
                y_train = np.copy(train_groundtruths[i])
            else:
                y_train = np.concatenate((y_train, train_groundtruths[i]))
                

        # One-hot encoding
        y_train = to_categorical(y_train)

        # Prepare testing data
        X_test = test_inputs[test_tag_i]
        y_test = test_groundtruths[test_tag_i]

        y_test = to_categorical(y_test)

        # Train/Test split for data
        print("Training data shape: (X) (y)")
        print(X_train.shape, y_train.shape)
        print("Testing data shape: (X) (y)")
        print(X_test.shape, y_test.shape)

        print('Data loaded! Ready to train')

        # repeat experiment
        scores = list()
        for r in range(repeats):
            score = build_model(X_train, y_train, X_test, y_test)
            score = score * 100.0
            print('>#%d: %.3f' % (r+1, score))
            scores.append(score)
        # summarize results
        m = summarize_results(scores)
        accuracies.append(m)

    print(f"OVERALL ACCURACY WAS {mean(accuracies)}")

run_exp()
