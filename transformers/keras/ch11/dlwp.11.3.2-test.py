print("Listing 11.5 Our model-building utility")
from tensorflow import keras
from tensorflow.keras import layers

max_tokens = 20000
inputs = keras.Input(shape=(max_tokens))
print("inputs.shape: ", inputs.shape)

inputs = keras.Input(shape=(max_tokens,))
print("inputs.shape: ", inputs.shape)

