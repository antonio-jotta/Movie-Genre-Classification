import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
from official.nlp import optimization
from sklearn.preprocessing import LabelEncoder


def train_model(train, validation):
    # Prepare the training data
    X_train = train['Plot'].tolist()
    y_train = train['Genre'].tolist()  # Make sure this is label-encoded

    # Prepare the validation data
    X_val = validation['Plot'].tolist()
    y_val = validation['Genre'].tolist()

    # Preprocess and BERT models from TensorFlow Hub
    preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    bert_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"

    # Define the classifier model
    def build_classifier_model(num_classes):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(preprocess_url, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(bert_url, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(num_classes, activation='softmax', name='classifier')(net)  # Changed here
        return tf.keras.Model(text_input, net)

    # Compile the model

    classifier_model = build_classifier_model(len(train['Genre'].unique()))

    # Step 2: Compile the model with an optimizer, loss function, and metrics
    # Example: Using Adam optimizer, SparseCategoricalCrossentropy loss for multi-class classification, and accuracy as the metric.
    classifier_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),  # Adam is commonly used with BERT
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])

    unique_genres = train['Genre'].unique()
    genre_to_index = {genre: idx for idx, genre in enumerate(unique_genres)}
    index_to_genre = {idx: genre for genre, idx in genre_to_index.items()}  # For decoding

    # Encode the labels using the dictionary
    y_train_encoded = np.array([genre_to_index[genre] for genre in y_train])
    y_val_encoded = np.array([genre_to_index[genre] for genre in y_val])

    # Use y_train_encoded for training
    history = classifier_model.fit(
        tf.constant(X_train), 
        y_train_encoded, 
        validation_data=(tf.constant(X_val), y_val_encoded), 
        epochs=3, 
        batch_size=16
    )

    classifier_model.save("trained_models/bert_en_uncased", save_format="tf")