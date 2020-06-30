import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_prepro():
    train_dir = "data/train"
    test_dir = "data/test"

    train_gen = ImageDataGenerator(rescale = (1.0/255))
    test_gen = ImageDataGenerator(rescale = (1.0/255))

    train_flow = train_gen.flow_from_directory(train_dir, target_size = (320,240))
    test_flow = test_gen.flow_from_directory(test_dir, target_size = (320,240))

    return train_flow, test_flow
