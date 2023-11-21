"""
Author: Gesang
Date: 16/11/2023
This is the sentiment_trainer.py module.
Usage:
- Trainer
"""

import os
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs


LABEL_KEY = "label"
FEATURE_KEY = "text"


def transformed_name(key):
    """
    Returns:
        tranformed key
    """

    return f"{key}_xf"


def gzip_reader_fn(filenames):
    """
    Returns:
        record dataset
    """
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64):
    """
    Returns:
        dataset
    """

    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset


def model_builder(vectorizer_layer):
    """
    Returns:
        model
    """

    inputs = tf.keras.Input(
        shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string
    )

    x = vectorizer_layer(inputs)
    x = layers.Embedding(input_dim=5000, output_dim=16)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation="relu")(x)

    outputs = layers.Dense(1, activation=tf.nn.sigmoid)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    model.summary()

    return model


def _get_serve_tf_example_fn(model, tf_transform_output):
    """
    Returns:
        example
    """

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)

        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec
        )
        transformed_features = model.tft_layer(parsed_features)

        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs):
    """
    Build Model
    """

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_binary_accuracy",
        mode="max",
        verbose=1,
        patience=2,
    )
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        fn_args.serving_model_dir,
        monitor="val_binary_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
    )
    callbacks = [
        tensorboard_callback,
        early_stopping_callback,
        model_checkpoint_callback
    ]

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = input_fn(
        fn_args.train_files, tf_transform_output, 10
    )
    eval_set = input_fn(
        fn_args.eval_files, tf_transform_output, 10
    )

    vectorizer_dataset = train_set.map(
        lambda f, l: f[transformed_name(FEATURE_KEY)]
    )

    vectorizer_layer = layers.TextVectorization(
        max_tokens=5000,
        output_mode="int",
        output_sequence_length=500,
    )
    vectorizer_layer.adapt(vectorizer_dataset)

    model = model_builder(vectorizer_layer)

    model.fit(
        x=train_set,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_set,
        validation_steps=fn_args.eval_steps,
        callbacks=callbacks,
        epochs=10,
        verbose=1,
    )

    signatures = {
        "serving_default": _get_serve_tf_example_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name="examples",
            )
        )
    }

    model.save(
        fn_args.serving_model_dir,
        save_format="tf",
        signatures=signatures
    )