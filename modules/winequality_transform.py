"""Modul preprocessing data untuk pipeline TFX wine quality."""

import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = 'quality'
NUMERIC_FEATURE_KEYS = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

def transformed_name(key):
    """Menambahkan akhiran _xf pada nama fitur."""
    return key + "_xf"

def preprocessing_fn(inputs):
    """Melakukan scaling scale_to_z_score pada fitur numerik dan konversi tipe data label.'"""
    outputs = {}
    
    for key in NUMERIC_FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.scale_to_z_score(inputs[key])
        
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY] - 3, tf.int64)
    
    return outputs
