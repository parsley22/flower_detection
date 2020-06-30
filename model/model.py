from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications.densenet import Densenet201


def build_model(input_shape, n_classes):

    pre_trained = Densenet201(input_shape = (input_shape), include_top = False, weights = None)
    for layer in pre_trained:
        layer.trainable = False
    
    last_layer = pre_trained.get_layer("mixed7")
    last_output = last_layer.output 

    x = layers.Flatten()(last_output)
    x = layers.Dense(1024, activation = 'relu')(X)
    x = layers.Dropout(0.2)(x)
    x - layers.Dense(n_classes, activation = 'softmax')(x)

    m = Model(pre_trained.input, x)
    m.compile(optimizer = Adam(), loss = CategoricalCrossentropy(), metrics = ['accuracy'])
    return m


