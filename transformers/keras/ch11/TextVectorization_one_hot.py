#
# https://towardsdatascience.com/building-a-one-hot-encoding-layer-with-tensorflow-f907d686bf39

# The experimental TextVectorization layer can be used to standardize 
# and tokenize sequences of strings, such as sentences, but for our use case, 
# we’ll simply convert individual string categories into integer indices.
import pandas as pd
from tensorflow.keras import layers

colors_df = pd.DataFrame(data=[['red'],['blue'],['green'],['blue']], columns=['color'])

text_vectorization = layers.experimental.preprocessing.TextVectorization(output_sequence_length=1)
text_vectorization.adapt(colors_df.values)

print('Red index:', text_vectorization.call([['red']]))
print('Blue index:', text_vectorization.call([['blue']]))
print('Green index:', text_vectorization.call([['green']]))

print(text_vectorization.get_vocabulary()) # prints [b'blue', b'red', b'green']

# The OneHotEncodingLayer Class
# Finally, we can now create the Class that will represent a One Hot Encoding Layer in a neural network.
class OneHotEncodingLayer(layers.experimental.preprocessing.PreprocessingLayer):
  def __init__(self, vocabulary=None, depth=None, minimum=None):
    super().__init__()
    self.vectorization = layers.experimental.preprocessing.TextVectorization(output_sequence_length=1)  

    if vocabulary:
      self.vectorization.set_vocabulary(vocabulary)
    self.depth = depth   
    self.minimum = minimum

  def adapt(self, data):
    self.vectorization.adapt(data)
    vocab = self.vectorization.get_vocabulary()
    self.depth = len(vocab)
    indices = [i[0] for i in self.vectorization([[v] for v in vocab]).numpy()]
    self.minimum = min(indices)

  def call(self,inputs):
    vectorized = self.vectorization.call(inputs)
    subtracted = tf.subtract(vectorized, tf.constant([self.minimum], dtype=tf.int64))
    encoded = tf.one_hot(subtracted, self.depth)
    return layers.Reshape((self.depth,))(encoded)

  def get_config(self):
    return {'vocabulary': self.vectorization.get_vocabulary(), 'depth': self.depth, 'minimum': self.minimum}

# Using the Custom Layer
# Now we can try the new layer out in a simple Neural Network.
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

colors_df = pd.DataFrame(data=[[5,'yellow'],[1,'red'],[2,'blue'],[3,'green'],[4,'blue'],[7,'purple']], columns=['id', 'color'])

categorical_input = layers.Input(shape=(1,), dtype=tf.string)
one_hot_layer = OneHotEncodingLayer()
one_hot_layer.adapt(colors_df['color'].values)
encoded = one_hot_layer(categorical_input)

numeric_input = layers.Input(shape=(1,), dtype=tf.float32)

concat = layers.concatenate([numeric_input, encoded])

model = models.Model(inputs=[numeric_input, categorical_input], outputs=[concat])
predicted = model.predict([colors_df['id'], colors_df['color']])
print(predicted)
# [[5. 0. 1. 0. 0. 0.]
#  [1. 0. 0. 1. 0. 0.]
#  [2. 1. 0. 0. 0. 0.]
#  [3. 0. 0. 0. 0. 1.]
#  [4. 1. 0. 0. 0. 0.]
#  [7. 0. 0. 0. 1. 0.]]

#
# Notice that it One Hot Encodes the color category in the same way as before, 
# so we know that the subsequent layers of our model will be provided the same 
# features in the same order as they appeared during training.
config = model.get_config()
with tf.keras.utils.custom_object_scope({'OneHotEncodingLayer': OneHotEncodingLayer}):
  loaded_model = tf.keras.Model.from_config(config)

predicted = model.predict([colors_df['id'], colors_df['color']])
print(predicted)
# [[5. 0. 1. 0. 0. 0.]
#  [1. 0. 0. 1. 0. 0.]
#  [2. 1. 0. 0. 0. 0.]
#  [3. 0. 0. 0. 0. 1.]
#  [4. 1. 0. 0. 0. 0.]
#  [7. 0. 0. 0. 1. 0.]]

#
# You can find a notebook containing all of the code examples here: 
# https://github.com/gnovack/tf-one-hot-encoder/blob/master/OneHotEncoderLayer.ipynb

