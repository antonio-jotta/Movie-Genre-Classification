import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np

def test_sample(test):
    X_test = test['Plot'].tolist()
    loaded_model = tf.keras.models.load_model('trained_models/bert_en_uncased')

    predictions = loaded_model.predict(tf.constant(X_test))
    return predictions
    