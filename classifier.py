import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


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
def classify(name):
    # load the dataset
    data = pd.read_csv(name)

    # shuffle the dataset
    data.sample(frac=1).reset_index()

    # save 20% of the data in lockbox
    # lockbox_split_index = int(0.95 * len(data))
    #
    # lockbox_x = data.iloc[lockbox_split_index:len(data):1, :]
    # lockbox_y = lockbox_x
    # lockbox_xf = lockbox_x.drop(columns=['annotation'])
    # lockbox_y = lockbox_y.loc[:, ['annotation']]
    # lockbox_yf = tf.keras.utils.to_categorical(lockbox_y, 20)
    #
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

    # creating a noise with the same dimension as the dataset (2,2)
    mu, sigma = 0, 0.2
    noise = np.random.normal(mu, sigma, x_trainf.shape)
    x_trainf += noise

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

    # test_model_lockbox(lockbox_xf, lockbox_yf)
    # test_name = 'insert name of dataset to test with model here'
    # test_model(test_name)


# (!#1)
