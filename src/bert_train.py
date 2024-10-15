import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_addons as tfa
import json

def train_model(train, validation):
    # Prepare the training data
    X_train = train['Plot'].tolist()
    y_train = train['Genre'].tolist()
    directors_train = train['Director'].tolist()

    # Prepare the validation data
    X_val = validation['Plot'].tolist()
    y_val = validation['Genre'].tolist()
    directors_val = validation['Director'].tolist()

    # Encode the director names as integer indices
    # Encode the director names as integer indices
    unique_directors = sorted(set(train['Director']).union(set(validation['Director'])))
    director_to_index = {director: idx for idx, director in enumerate(unique_directors)}
    director_to_index["<UNK>"] = len(director_to_index) # Spot for OOV words

    # Save the mapping for later use in testing
    with open('director_to_index.json', 'w') as f:
        json.dump(director_to_index, f)

    directors_train_encoded = np.array([
        director_to_index.get(director, director_to_index["<UNK>"]) for director in directors_train
    ], dtype=np.int32)

    directors_val_encoded = np.array([
        director_to_index.get(director, director_to_index["<UNK>"]) for director in directors_val
    ], dtype=np.int32)
    
    # Preprocess and BERT models from TensorFlow Hub
    preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    bert_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"

    # Define the classifier model
    def build_classifier_model(num_classes, embedding_dim=16):
        # Text input and BERT processing
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(preprocess_url, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(bert_url, trainable=True, name='BERT_encoder')
        bert_outputs = encoder(encoder_inputs)
        bert_output = bert_outputs['pooled_output']

        # Director input and embedding layer
        director_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='director')
        director_embedding = tf.keras.layers.Embedding(input_dim=len(director_to_index)+1, output_dim=embedding_dim, trainable=False)(director_input)
        director_embedding = tf.keras.layers.Flatten()(director_embedding)

        # Concatenate BERT output with director embedding
        concatenated = tf.keras.layers.Concatenate()([bert_output, director_embedding])

        # Final dense layers
        net = tf.keras.layers.Dropout(0.1)(concatenated)
        net = tf.keras.layers.Dense(num_classes, activation='softmax', name='classifier')(net)
        return tf.keras.Model([text_input, director_input], net)

    # Build and compile the model
    num_classes = len(train['Genre'].unique())
    classifier_model = build_classifier_model(num_classes)

    # Convert genres to integer indices with dtype int32
    unique_genres = train['Genre'].unique()
    genre_to_index = {genre: idx for idx, genre in enumerate(unique_genres)}
    
    y_train_encoded = np.array([genre_to_index[genre] for genre in y_train], dtype=np.int32)
    y_val_encoded = np.array([genre_to_index[genre] for genre in y_val], dtype=np.int32)

    classifier_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                            metrics=['accuracy'])

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2)

    # Train the model with validation data
    classifier_model.fit(
        [tf.constant(X_train), directors_train_encoded],
        y_train_encoded,
        validation_data=([tf.constant(X_val), directors_val_encoded], y_val_encoded),
        epochs=10,
        callbacks=[early_stopping, reducelr]
    )

    classifier_model.save('trained_models/bert_en_uncased')
