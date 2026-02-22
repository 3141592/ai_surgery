# Suppress warnings
import os, pathlib
from ai_surgery.data_paths import get_data_root

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Force CPU use for keras.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

DATA_ROOT = get_data_root() / "aclImdb"

MODEL_PATH = (
    get_data_root()
    / "models"
    / "one_hot_bidir_gru.keras"
)

print("11.3.3 Processing words as a sequence: The sequence model approach")
import tensorflow as tf
from tensorflow import keras
batch_size = 16

train_ds = keras.utils.text_dataset_from_directory(
                DATA_ROOT / "train/",
                batch_size=batch_size)

val_ds = keras.utils.text_dataset_from_directory(
                DATA_ROOT / "val/",
                batch_size=batch_size)

test_ds = keras.utils.text_dataset_from_directory(
                DATA_ROOT / "test/", 
                batch_size=batch_size)

text_only_train_ds = train_ds.map(lambda x, y: x)

print("Listing 11.12 Preparing integer sequence datasets")
from tensorflow.keras import layers

max_length = 600
max_tokens = 20000
text_vectorization = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        # In order to keep a manageable input size, we'll truncate the inputs after the first 600 words.
        output_sequence_length=max_length,
)
text_vectorization.adapt(text_only_train_ds)

int_train_ds = train_ds.map(
                lambda x, y: (text_vectorization(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)
int_val_ds = val_ds.map(
                lambda x, y: (text_vectorization(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)
int_test_ds = test_ds.map(
                lambda x, y: (text_vectorization(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)

print("11.16 Model that uses an Embedding layer trained from scratch")
inputs = keras.Input(shape=(None,), dtype="int64")
embedded = layers.Embedding(input_dim=max_tokens, output_dim=256)(inputs)
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=["accuracy"])
model.summary()

callbacks = [
        keras.callbacks.ModelCheckpoint(MODEL_PATH,
            save_best_only=True)
]
model.fit(int_train_ds,
        validation_data=int_val_ds,
        epochs=10,
        callbacks=callbacks)
model = keras.models.load_model(MODEL_PATH)
print("Evaluating best checkpoint on test set")
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")

print("# --- Embedding exploration: trained-from-scratch embedding ---")

import numpy as np

print("# 1 Get the vocabulary and build word -> index mapping")
vocab = text_vectorization.get_vocabulary()  # list where index = token id
word_to_idx = {w: i for i, w in enumerate(vocab)}

print(f"Vocab size from TextVectorization: {len(vocab):,}")
print("Special tokens (usually):", vocab[:5])  # often: ['', '[UNK]', ...]

print("# 2 Extract embedding matrix W: shape (max_tokens, 256)")
# Find the Embedding layer (safer than assuming layer index)
emb_layer = next((l for l in model.layers if isinstance(l, keras.layers.Embedding)), None)
if emb_layer is None:
    raise RuntimeError("Could not find an Embedding layer in the loaded model.")
W = emb_layer.get_weights()[0]
print("Embedding matrix shape:", W.shape)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float | None:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return None
    return float(np.dot(v1, v2) / (n1 * n2))

def get_word_vector(word: str) -> np.ndarray | None:
    idx = word_to_idx.get(word)
    if idx is None:
        return None
    if idx >= W.shape[0]:
        return None
    return W[idx]

def get_closest_words(word: str, top_k: int = 10) -> None:
    print(f"\nGet top {top_k} words similar to '{word}' (scratch embedding):")

    v = get_word_vector(word)
    if v is None:
        print("Word not found in TextVectorization vocabulary.")
        return

    sims: list[tuple[str, float]] = []
    for w, idx in word_to_idx.items():
        # Skip special tokens and the query word
        if w in {"", "[UNK]"} or w == word:
            continue
        if idx >= W.shape[0]:
            continue

        sim = cosine_similarity(v, W[idx])
        if sim is None:
            continue

        sims.append((w, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    for w, sim in sims[:top_k]:
        print(f"{w}: {sim:.4f}")

# Try a few probes (pick words that actually appear in IMDB)
get_closest_words("good", top_k=10)
get_closest_words("bad", top_k=10)
get_closest_words("great", top_k=10)
get_closest_words("terrible", top_k=10)