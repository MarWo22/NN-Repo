import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import signal

# ------------------------------------------------------------------------------------------------------------------#
#   Call the script as 'neural_network.py "path_to_timestamp.csv" "path_to_annotations.txt"'                        #
# ------------------------------------------------------------------------------------------------------------------#


#
# This part of the script will handle all the data preprocessing
#

def read_data(dataPath):
    data = pd.read_csv(dataPath, usecols=[1,2])
    return data


# Reads the annotations from the .txt file
def read_annotations(annotationsPath):
    data = pd.read_csv(annotationsPath, delim_whitespace=True, usecols=['Sample', '#'])
    data = data.drop(0).reset_index(drop=True) #drop first row
    return data


# Applies a 3rd order butterworth filters with a 0.5 low cutoff and 45 high cutoff
def apply_filter(dataframe):
    nyq = 0.5 * 360
    low = 0.4 / nyq
    high = 45 / nyq
    sos = signal.butter(3, [low, high], analog=False, btype='band', output='sos')
    dataframe[dataframe.columns[0]] = signal.sosfilt(sos, dataframe[dataframe.columns[0]])
    dataframe[dataframe.columns[1]] = signal.sosfilt(sos, dataframe[dataframe.columns[1]])
    return dataframe


# Creates a static 160 timestamp window of each heartbeat
def create_static_windows(annotations_dataframe, timeseries_dataframe):

    data = []
    for i in range(1, len(annotations_dataframe.index)):
        current_annotations = annotations_dataframe.values[i]
        low_range = current_annotations[0]-40
        high_range = current_annotations[0]+40
        if low_range >= 0 and high_range < len(timeseries_dataframe): # makes sure the window doesnt go out of bounds
            timeseries_window = list(timeseries_dataframe[timeseries_dataframe.columns[0]][low_range:high_range]) + list(timeseries_dataframe[timeseries_dataframe.columns[1]][low_range:high_range])
            data.append([current_annotations[1], current_annotations[2]] + timeseries_window)
    
    headers = ['annotation', 'time_since_last_beat'] + ["timestamp_" + str(i) for i in range(0, 160)]
    dataframe = pd.DataFrame(np.asarray(data), columns=headers)
    for column in dataframe.columns[1:]:
        dataframe = dataframe.astype({column:float})

    return dataframe


# Rebalances the data to make sure that the frequency of normal and pathological heartbeats are the same
def rebalance(df):

    normal_beats = df[df['annotation'] == 0]
    pathological_beats = df[df['annotation'] != 0]

    if len(normal_beats) > 0 and len(pathological_beats) > 0 and len(normal_beats) < len(pathological_beats):
        new_sample = normal_beats.sample(len(pathological_beats), replace=True)
        df = pd.concat([new_sample, pathological_beats], axis=0)
    elif len(normal_beats) > 0 and len(pathological_beats) > 0 and len(normal_beats) > len(pathological_beats):
        new_sample = pathological_beats.sample(len(normal_beats), replace=True)
        df = pd.concat([new_sample, normal_beats], axis=0)
    return df.sample(frac=1).reset_index(drop=True) #shuffle

# Normalizes a dataset
def normalize_dataframe(df):
    timeSinceLastBeat = df.iloc[:,0]
    timeseries = df.iloc[:, 1:]
    timeseries = timeseries.apply(lambda x: (x-x.min())/ (x.max()-x.min()), axis=0)
    timeseries.insert(0, 'time_since_last_beat', timeSinceLastBeat)
    return timeseries



# Replaces the characters by digits to prepare for one-hot-encoding, and removes any rows that contain characters that are not valid annotations
def replace_annotations(dataframe):
    dataframe = dataframe[dataframe.annotation.isin(['N', 'L', 'R', 'A', 'a', 'J', 'S', 'V', 'F', '[', '!', ']', 'e', 'j', 'E', '/', 'f', 'x', 'Q', '|'])]
    dataframe['annotation'] = dataframe['annotation'].replace({'N':0, 'L':1, 'R':2, 'A':3, 'a':4, 'J':5, 'S':6, 'V':7, 'F':8, '[':9, '!':10, ']':11, 'e':12, 'j':13, 'E':14, '/':15, 'f':16, 'x':17, 'Q':18, '|':19})
    return dataframe


# Get the number of timesteps that have passed since the last heartbeat
def get_time_since_last_beat(annotations_dataframe):
    time_since_last_beat = []
    for i in range(1, len(annotations_dataframe.index)):
        time_since_last_beat.append(annotations_dataframe.values[i][0] - annotations_dataframe.values[i - 1][0])
    annotations_dataframe = annotations_dataframe.drop(0).reset_index(drop=True)
    annotations_dataframe["timeSinceLastBeat"] = time_since_last_beat # The first heartbeat needs to be dropped since it has no previous beat
    return annotations_dataframe


def preprocess(timeseries_path, annotations_path):

    timestamp_dataframe = read_data(timeseries_path)
    annotations_dataframe = read_annotations(annotations_path)
    timestamp_dataframe = apply_filter(timestamp_dataframe)
    annotations_dataframe = get_time_since_last_beat(annotations_dataframe)
    dataframe = create_static_windows(annotations_dataframe, timestamp_dataframe)
    dataframe = replace_annotations(dataframe)
    annotations = dataframe.iloc[:,0] # remove the annotations column during normalization
    dataframe = normalize_dataframe(dataframe.iloc[:,1:])
    dataframe.insert(0, 'annotation', annotations) # adding it back after normalization is finished
    balanced_dataframe = rebalance(dataframe)
    return balanced_dataframe








#
# This part of the script will handle the actual modeling
#

# plots the accuracy and loss history for the network
def plot_history(history):
    figure, axes = plt.subplots(2)

    # plot training and testing accuracy
    axes[0].plot(history.history['accuracy'], label="train accuracy")
    axes[0].plot(history.history['val_accuracy'], label="test accuracy")
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(loc='lower right')
    axes[0].set_title('Accuracy history')

    # plot training and testing loss
    axes[1].plot(history.history['loss'], label="train error")
    axes[1].plot(history.history['val_loss'], label="test error")
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Error')
    axes[1].legend(loc='upper right')
    axes[1].set_title('Error history')

    plt.show()


# can be used to test the best model with other datasets
def test_model(name):
    best_model = tf.keras.models.load_model('my_best_model.h5')
    test_dataset = pd.read_csv(name)
    x_testf = test_dataset.drop(columns=['annotation'])
    y_test = test_dataset.loc[:, ['annotation']]
    y_testf = tf.keras.utils.to_categorical(y_test, 20)

    score = best_model.evaluate(x_testf, y_testf, verbose=1)
    print("%s: %.2f%%" % (best_model.metrics_names[2], score[2] * 100))


# can be used to test the best model with locked dataset
def test_model_lockbox(x_testf, y_testf):
    best_model = tf.keras.models.load_model('my_best_model.h5')

    score = best_model.evaluate(x_testf, y_testf, verbose=1)
    print("%s: %.2f%%" % (best_model.metrics_names[2], score[2] * 100))


# trains and validates the network on the given dataset
def model(data):

    # save 20% of the data in lockbox
   # lockbox_split_index = int(0.95 * len(data))
#
    #lockbox_x = data.iloc[lockbox_split_index:len(data):1, :]
    #lockbox_y = lockbox_x
   # lockbox_xf = lockbox_x.drop(columns=['annotation'])
   # lockbox_y = lockbox_y.loc[:, ['annotation']]
   # lockbox_yf = tf.keras.utils.to_categorical(lockbox_y, 20)

   # data = data.iloc[0:lockbox_split_index:1, :]

    # split remaining dataframe into training and testing sets with an 70-30 split
    split_index = int(0.2 * len(data))

    x_train = data.iloc[split_index:len(data):1, :]
    y_train = x_train
    x_trainf = x_train.drop(columns=['annotation'])
    y_train = y_train.loc[:, ['annotation']]

    x_validation = data.iloc[0:split_index:1, :]
    y_validation = x_validation
    x_validationf = x_validation.drop(columns=['annotation'])
    y_validation = y_validation.loc[:, ['annotation']]

    # sdding noise to the training input with a standard deviation of 0.05
    mu, sigma = 0, 0.05
    noise = np.random.normal(mu, sigma, x_trainf.shape)
    x_trainf += noise
    # normalizing again after adding the noise

    # convert labels to categorical data
    y_trainf = tf.keras.utils.to_categorical(y_train, 20)
    y_validationf = tf.keras.utils.to_categorical(y_validation, 20)

    # initialize model
    model = tf.keras.Sequential()

    # define model architecture
    model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(x_trainf.shape[1],),
                                    kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    model.add(tf.keras.layers.Dense(64, activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    model.add(tf.keras.layers.Dropout(0.05))
    
    model.add(tf.keras.layers.Dense(20, activation='softmax'))

    # model compilation
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['mse', 'accuracy']
    )

    # initialize model callback to save best model
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='my_best_model.h5',
        monitor='mse',
        save_best_only=True,
        mode='max')

    # initialize early stopping
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='mse',
                                                               patience=20)

    # initialize logging metrics with tensorboard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

    # train the model
    history = model.fit(
        x_trainf, y_trainf,
        batch_size=128,
        epochs=200,
        validation_data=(x_validationf, y_validationf),
        callbacks=[checkpoint_callback, early_stopping_callback,
                   tensorboard_callback],
    )

    plot_history(history)

    #test_model_lockbox(lockbox_xf, lockbox_yf)
    #test_name = 'insert name of dataset to test with model here'
    #test_model(test_name)


# (!#1)



if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Invalid arguments")
        exit(0)
    preprocessed_data = preprocess(sys.argv[1], sys.argv[2])
    model(preprocessed_data)
