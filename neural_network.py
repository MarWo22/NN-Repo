import pandas as pd
import tensorflow as tf
import keras_tuner as kt
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
    data = pd.read_csv(dataPath, usecols=[1, 2])
    return data


# Reads the annotations from the .txt file
def read_annotations(annotationsPath):
    data = pd.read_csv(annotationsPath, delim_whitespace=True, usecols=['Sample', '#'])
    data = data.drop(0).reset_index(drop=True)  # drop first row
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
        low_range = current_annotations[0] - 40
        high_range = current_annotations[0] + 40
        if low_range >= 0 and high_range < len(timeseries_dataframe):  # makes sure the window doesnt go out of bounds
            timeseries_window = list(
                timeseries_dataframe[timeseries_dataframe.columns[0]][low_range:high_range]) + list(
                timeseries_dataframe[timeseries_dataframe.columns[1]][low_range:high_range])
            data.append([current_annotations[1], current_annotations[2]] + timeseries_window)

    headers = ['annotation', 'time_since_last_beat'] + ["timestamp_" + str(i) for i in range(0, 160)]
    dataframe = pd.DataFrame(np.asarray(data), columns=headers)
    for column in dataframe.columns[1:]:
        dataframe = dataframe.astype({column: float})

    return dataframe


# Rebalances the data to make sure that the frequency of normal and pathological heartbeats are the same
def rebalance(df):
    normal_beats = df[df['annotation'] == 0]
    pathological_beats = df[df['annotation'] != 0]

    if 0 < len(normal_beats) < len(pathological_beats) and len(pathological_beats) > 0:
        new_sample = normal_beats.sample(len(pathological_beats), replace=True)
        df = pd.concat([new_sample, pathological_beats], axis=0)
    elif len(normal_beats) > 0 and 0 < len(pathological_beats) < len(normal_beats):
        new_sample = pathological_beats.sample(len(normal_beats), replace=True)
        df = pd.concat([new_sample, normal_beats], axis=0)
    return df.sample(frac=1).reset_index(drop=True)  # shuffle


# Normalizes a dataset
def normalize_dataframe(df):
    timeSinceLastBeat = df.iloc[:, 0]
    timeseries = df.iloc[:, 1:]
    timeseries = timeseries.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    timeseries.insert(0, 'time_since_last_beat', timeSinceLastBeat)
    return timeseries


# Replaces the characters by digits to prepare for one-hot-encoding, and removes any rows that contain characters
# that are not valid annotations
def replace_annotations(dataframe):
    dataframe = dataframe[dataframe.annotation.isin(
        ['N', 'L', 'R', 'A', 'a', 'J', 'S', 'V', 'F', '[', '!', ']', 'e', 'j', 'E', '/', 'f', 'x', 'Q', '|'])]
    dataframe['annotation'] = dataframe['annotation'].replace(
        {'N': 0, 'L': 1, 'R': 2, 'A': 3, 'a': 4, 'J': 5, 'S': 6, 'V': 7, 'F': 8, '[': 9, '!': 10, ']': 11, 'e': 12,
         'j': 13, 'E': 14, '/': 15, 'f': 16, 'x': 17, 'Q': 18, '|': 19})
    return dataframe


# Get the number of timesteps that have passed since the last heartbeat
def get_time_since_last_beat(annotations_dataframe):
    time_since_last_beat = []
    for i in range(1, len(annotations_dataframe.index)):
        time_since_last_beat.append(annotations_dataframe.values[i][0] - annotations_dataframe.values[i - 1][0])
    # The first heartbeat needs to be dropped since it has no previous beat
    annotations_dataframe = annotations_dataframe.drop(0).reset_index(drop=True)
    annotations_dataframe["timeSinceLastBeat"] = time_since_last_beat
    return annotations_dataframe


def preprocess(timeseries_path, annotations_path):
    timestamp_dataframe = read_data(timeseries_path)
    annotations_dataframe = read_annotations(annotations_path)
    timestamp_dataframe = apply_filter(timestamp_dataframe)
    annotations_dataframe = get_time_since_last_beat(annotations_dataframe)
    dataframe = create_static_windows(annotations_dataframe, timestamp_dataframe)
    dataframe = replace_annotations(dataframe)
    annotations = dataframe.iloc[:, 0]  # remove the annotations column during normalization
    dataframe = normalize_dataframe(dataframe.iloc[:, 1:])
    dataframe.insert(0, 'annotation', annotations)  # adding it back after normalization is finished
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


# splits the processed data into train and validation sets and adds noise
def split_data(data):
    # split remaining dataframe into training and testing sets with an 80-20 split
    split_index = int(0.2 * len(data))

    x_train = data.iloc[split_index:len(data):1, :]
    y_train = x_train
    x_trainfinal = x_train.drop(columns=['annotation'])
    y_train = y_train.loc[:, ['annotation']]

    x_validation = data.iloc[0:split_index:1, :]
    y_validation = x_validation
    x_validationfinal = x_validation.drop(columns=['annotation'])
    y_validation = y_validation.loc[:, ['annotation']]

    # sdding noise to the training input with a standard deviation of 0.05
    mu, sigma = 0, 0.05
    noise = np.random.normal(mu, sigma, x_trainfinal.shape)
    x_trainfinal += noise

    # convert labels to categorical data
    y_trainfinal = tf.keras.utils.to_categorical(y_train, 20)
    y_validationfinal = tf.keras.utils.to_categorical(y_validation, 20)

    return x_trainfinal, y_trainfinal, x_validationfinal, y_validationfinal


# builds the model with given hyperparameters
def build_model(hp):
    # initialize model
    model = tf.keras.Sequential()

    # initialize hyperparameter tuning
    hp_units1 = hp.Int('units1', min_value=32, max_value=512, step=32)
    hp_units2 = hp.Int('units2', min_value=32, max_value=512, step=32)
    hp_reg_lr1 = hp.Choice('reg_learning_rate1', [1e-2, 1e-3, 1e-4])
    hp_reg_lr2 = hp.Choice('reg_learning_rate2', [1e-2, 1e-3, 1e-4])
    hp_opt_lr = hp.Choice('opt_learning_rate', [1e-3, 1e-4, 1e-5])
    hp_drop_rate = hp.Choice('drop_rate', [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])

    # define model architecture
    model.add(tf.keras.layers.Dense(units=hp_units1, activation='relu', input_shape=(161,),
                                    kernel_regularizer=tf.keras.regularizers.l2(hp_reg_lr1)))

    model.add(tf.keras.layers.Dropout(hp_drop_rate))

    model.add(tf.keras.layers.Dense(hp_units2, activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(hp_reg_lr2)))

    model.add(tf.keras.layers.Dense(20, activation='softmax'))

    # initialize the adam optimizer
    optimizer = tf.keras.optimizers.Adam(hp_opt_lr)

    # model compilation
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['mse', 'accuracy']
    )

    return model


# tune the hyperparameters to get the best model
def tune_model(x_train, y_train, x_validation, y_validation):
    tuner = kt.Hyperband(build_model,
                         objective='val_accuracy',
                         max_epochs=200,
                         factor=3,
                         directory='my_dir',
                         project_name='NN-Repo')

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='mse', patience=20)

    tuner.search(x_train, y_train,
                 epochs=200,
                 validation_data=(x_validation, y_validation),
                 callbacks=[early_stopping_callback],
                 verbose=0)

    best_params = tuner.get_best_hyperparameters(num_trials=1)[0]

    model = tuner.hypermodel.build(best_params)

    print(f"""
    Neurons in first layer = {best_params.get('units1')}.\n
    Neurons in second layer = {best_params.get('units2')}.\n
    First reg lr = {best_params.get('reg_learning_rate1')}.\n
    Second reg lr = {best_params.get('reg_learning_rate2')}.\n
    Optimizer lr = {best_params.get('op_learning_rate')}.\n
    Dropout rate = {best_params.get('drop_rate')}.
    """)

    model.summary()

    return model


# train the dataset on the best model
def train_model(model, x_trainfbest, y_trainfbest, x_validationfbest, y_validationfbest):
    # initialize model callback to save best model
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='my_best_model.h5',
        monitor='mse',
        save_best_only=True,
        mode='max')

    # initialize early stopping
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='mse', patience=20)

    # initialize logging metrics with tensorboard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

    # train the model
    history = model.fit(
        x_trainfbest, y_trainfbest,
        batch_size=128,
        epochs=200,
        validation_data=(x_validationfbest, y_validationfbest),
        callbacks=[checkpoint_callback, early_stopping_callback,
                   tensorboard_callback],
    )

    plot_history(history)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Invalid arguments")
        exit(0)
    preprocessed_data = preprocess(sys.argv[1], sys.argv[2])
    x_trainf, y_trainf, x_validationf, y_validationf = split_data(preprocessed_data)
    best_model = tune_model(x_trainf, y_trainf, x_validationf, y_validationf)
    train_model(best_model, x_trainf, y_trainf, x_validationf, y_validationf)
