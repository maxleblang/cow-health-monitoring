from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, GlobalAveragePooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D, Conv2D
from tensorflow.keras.layers import MaxPooling1D, MaxPooling2D, AveragePooling1D
from tensorflow.keras.layers import concatenate


'''
Pure CNN model from project
Runs two convolutions on the data
'''
def CNN(n_timesteps, n_features, n_outputs):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=4, activation='relu',
              input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=32, kernel_size=4, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model

'''
Multi-headed CNN
Accel data is data 1
UWB is data 2
'''
def multihead_CNN(n_timesteps, n1_features, n2_features, n_outputs):
    # head 1: for accel data
    inputs1 = Input(shape=(n_timesteps, n1_features))
    conv10 = Conv1D(filters=32, kernel_size=4, activation='relu')(inputs1)
    conv11 = Conv1D(filters=32, kernel_size=4, activation='relu')(conv10)
    pool10 = MaxPooling1D(pool_size=2)(conv11)
    drop10 = Dropout(0.2)(pool10)
    conv12 = Conv1D(filters=64, kernel_size=4, activation='relu')(drop10)
    conv13 = Conv1D(filters=64, kernel_size=4, activation='relu')(conv12)
    pool11 = MaxPooling1D(pool_size=2)(conv13)
    drop11 = Dropout(0.2)(pool11)
    gap1 = GlobalAveragePooling1D()(drop11)

    # head 2: for uwb data
    inputs2 = Input(shape=(n_timesteps, n2_features))
    '''
    conv20 = Conv1D(filters=32, kernel_size=4, activation='relu')(inputs2)
    conv21 = Conv1D(filters=32, kernel_size=4, activation='relu')(conv20)
    pool20 = MaxPooling1D(pool_size=2)(conv21)
    drop20 = Dropout(0.2)(pool20)
    conv22 = Conv1D(filters=64, kernel_size=4, activation='relu')(drop20)
    conv23 = Conv1D(filters=64, kernel_size=4, activation='relu')(conv22)
    pool21 = MaxPooling1D(pool_size=2)(conv23)
    drop21 = Dropout(0.2)(pool21)
    '''
    gap2 = GlobalAveragePooling1D()(inputs2)
        
    # merge
    merged = concatenate([gap1, gap2])
    # interpretation
    dense1 = Dense(100, activation='relu')(merged)
    outputs = Dense(n_outputs, activation='softmax')(dense1)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy', f1_m])

    return model
    


def CNN2D(wv_x, wv_y, wv_channels, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                    activation='relu',
                    input_shape=(wv_x, wv_y, wv_channels)))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model


# Evalutation metrics
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))