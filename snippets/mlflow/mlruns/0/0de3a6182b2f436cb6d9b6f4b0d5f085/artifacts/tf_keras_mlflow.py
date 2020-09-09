import tensorflow as tf
import numpy as np
import mlflow
import os



EPOCHS = 50
BATCH = 32
OPTIMIZER = 'adam'
LOSS = 'mse'
VALID_SPLIT = 0.2


def mlflow_model_summary(my_model, path='model_summary.txt'):
    # show model setup in artifacts
    with open(path, 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        my_model.summary(print_fn=lambda x: fh.write(x + '\n'))
    mlflow.log_artifact(path)
    return


class MlflowCallBacks(tf.keras.callbacks.Callback):

    # initialize additional variables
    def __init__(self):
        self.checkpoint_path = "model_checkpoint/"
        # set path
        if not os.path.isdir(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        self.config_path = self.checkpoint_path + "configuration_model.json"

    def on_train_begin(self, logs=None):
        # save model configuration
        model_json = self.model.to_json()
        with open(self.config_path, "w") as json_file:
            json_file.write(model_json)
        # log file to mlflow
        mlflow.log_artifact(local_path=self.checkpoint_path)
        mlflow.set_tag("Current epoch", 0)
        return

    def on_train_end(self, logs=None):
        # serialize weights to HDF5
        self.model.save_weights(self.checkpoint_path + "final_epoch_model_weights.hdf5")
        # log file to mlflow
        mlflow.log_artifact(local_path=self.checkpoint_path)
        mlflow.set_tag("Current epoch", "Done!")
        return

    def on_epoch_end(self, epoch, logs={}):
        offset_epoch = epoch + 1
        # log each metric to mlflow
        [mlflow.log_metric(key=name, value=value, step=offset_epoch) for name, value in logs.items()]
        # save weights at each epoch
        tmp_path = self.checkpoint_path + str(offset_epoch) + "_epoch_model_weights.hdf5"
        self.model.save_weights(tmp_path)
        # log file to mlflow
        mlflow.log_artifact(local_path=self.checkpoint_path)
        mlflow.set_tag("Current epoch", offset_epoch)
        return

# random data
x = np.random.random((100, 3))
y = np.random.random((100, 1))
print(x.shape, y.shape)

# model
inputs = tf.keras.Input(shape=(3))
layer = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(layer)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=OPTIMIZER, loss=LOSS)
print(model.summary())

# mlflow logs and train model
with mlflow.start_run():
    # log tag
    mlflow.set_tag("version", "tf-keras")

    # log parameters
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch", BATCH)
    mlflow.log_param("valid_split", VALID_SPLIT)
    mlflow.log_param("optimizer", OPTIMIZER)
    mlflow.log_param("loss", LOSS)

    # log artifacts
    mlflow.log_artifact("tf_keras_mlflow.py")
    mlflow_model_summary(my_model=model)

    # train model with callbacks
    model.fit(x, y, validation_split=VALID_SPLIT, epochs=EPOCHS, batch_size=BATCH, callbacks=[MlflowCallBacks()])

# see model configuration
print(model.get_config())