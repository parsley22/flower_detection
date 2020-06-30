from model import model, data_load
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

# Ensure keras is using GPU

import keras.backend.tensorflow_backend as tfback

print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
config

train_flow, test_flow = data_load.build_prepro()

m = model.build_model((320,240,3),5)

history = m.fit(train_flow,validation_data = test_flow, epochs = 25)

m.save_weights("model_weights")

plt.plot(history.history['accuracy'], label = "train accuracy")
plt.plot(history.history['val_accuracy'], label = "validation accuracy")
plt.plot(history.history['loss'], label = "train loss")
plt.plot(history.history['val_loss'], label = "validation loss")
plt.legend()
plt.savefig("plot.png")
plt.show()
