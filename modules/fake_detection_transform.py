"""
Author: Gesang
Date: 16/11/2023
This is the sentiment_transform.py module.
Usage:
- Transform
"""

import tensorflow as tf

LABEL_KEY = "label"
FEATURE_KEY = "text"


def transformed_name(key):
    """
    Returns:
        tranformed key
    """

    return f"{key}_xf"


def preprocessing_fn(inputs):
    """
    Returns:
        output preprocessing
    """

    outputs = {}

    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(
        inputs[FEATURE_KEY]
    )
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
