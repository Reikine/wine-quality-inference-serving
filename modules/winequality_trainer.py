"""Trainer module for wine quality model"""

import os

import tensorflow as tf
import keras_tuner as kt
import tensorflow_transform as tft
from keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult
from keras.utils import plot_model

from modules.winequality_transform import (
    LABEL_KEY,
    NUMERIC_FEATURE_KEYS,
    transformed_name,
)

def _gzip_reader_fn(filenames):
    """Memuat data dalam format TFRecord."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _input_fn(file_pattern, tf_transform_output, batch_size=32):
    """Menghasilkan fitur dan label untuk proses tuning/training.
    Args:
        file_pattern: pola berkas tfrecord masukan.
        tf_transform_output: Objek TFTransformOutput.
        batch_size: merepresentasikan jumlah elemen berurutan dari 
        dataset yang dikembalikan untuk digabungkan dalam satu batch.
    Returns:
        Sebuah dataset yang berisi tuple (fitur, indeks) di mana fitur
        adalah dictionary dari Tensor, dan indeks adalah Tensor tunggal
        dari indeks label.
    """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset

def model_builder(hp, transform_output):
    """Membangun dan mengompilasi model Keras dengan hyperparameter yang dinamis.

    Args:
        hp (kt.HyperParameters): Objek KerasTuner untuk mendefinisikan ruang 
        lingkup pencarian hyperparameter.
        transform_output (tft.TFTransformOutput): Objek keluaran dari proses 
        Transform yang memuat spesifikasi fitur.

    Returns:
        tf.keras.Model: Objek model Keras yang telah dikompilasi dan siap dilatih.
    """
    hp_units_1 = hp.Int('units_1', min_value=32, max_value=128, step=32)
    hp_units_2 = hp.Int('units_2', min_value=16, max_value=64, step=16)
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3])

    feature_spec = transform_output.transformed_feature_spec().copy()
    feature_spec.pop(transformed_name(LABEL_KEY))

    # Membuat input layer untuk setiap fitur.
    inputs = {
        key: tf.keras.Input(
            shape=(1,),
            name=key,
        )
        for key in feature_spec.keys()
    }

    x = layers.Concatenate()(list(inputs.values()))
    x = layers.Dense(hp_units_1, activation='relu')(x)
    x = layers.Dense(hp_units_2, activation='relu')(x)
    outputs = layers.Dense(6, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
        metrics=['sparse_categorical_accuracy'],
    )

    model.summary()

    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Membangun fungsi inferensi model untuk menangani data mentah saat tahap produksi.

    Args:
        model (tf.keras.Model): Objek model Keras yang telah dilatih.
        tf_transform_output (tft.TFTransformOutput): Objek keluaran dari proses 
        Transform untuk memuat layer transformasi data.

    Returns:
        Callable: Fungsi yang didekorasi dengan tf.function untuk memproses 
        contoh data mentah (serialized tf.examples) hingga menghasilkan output prediksi.
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
        
        outputs = model(transformed_features)
        
        return {'outputs': outputs}

    return serve_tf_examples_fn

def tuner_fn(fn_args: FnArgs):
    """Mengeksekusi proses pencarian hyperparameter terbaik untuk model.

    Args:
        fn_args (FnArgs): Objek berisi argumen konfigurasi seperti path file, 
        working_dir, train_steps, dan eval_steps.

    Returns:
        TunerFnResult: Objek yang berisi instance Tuner dan parameter fit_kwargs 
        untuk proses pelatihan.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = _input_fn(
        fn_args.train_files, tf_transform_output, batch_size=32
    )
    val_set = _input_fn(
        fn_args.eval_files, tf_transform_output, batch_size=32
    )

    tuner = kt.RandomSearch(
        hypermodel=lambda hp: model_builder(hp, tf_transform_output),
        objective="val_sparse_categorical_accuracy",
        max_trials=5,
        directory=fn_args.working_dir,
        project_name="wine_quality_tuning",
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_set,
            "validation_data": val_set,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
        },
    )

def run_fn(fn_args: FnArgs) -> None:
    """Mengeksekusi proses pelatihan model menggunakan hyperparameter terbaik,
    menyimpan visualisasi arsitektur model, dan mengekspor model untuk serving.

    Args:
        fn_args (FnArgs): Objek berisi argumen konfigurasi pelatihan seperti 
        train_files, eval_files, transform_graph_path, dan serving_model_dir.
    """
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'log')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq='batch'
    )

    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_sparse_categorical_accuracy',
        mode='max',
        verbose=1,       
        patience=10
    )

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = _input_fn(fn_args.train_files, tf_transform_output)
    val_set = _input_fn(fn_args.eval_files, tf_transform_output)

    if fn_args.hyperparameters:
        hp = kt.HyperParameters.from_config(fn_args.hyperparameters)
    else:
        hp = kt.Hyperparameters()

    model = model_builder(hp, tf_transform_output)

    model.fit(
        x=train_set,
        validation_data=val_set,
        callbacks=[tensorboard_callback, es],
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        epochs=10,
    )

    signatures = {
        'serving_default': _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'
            )
        )
    }

    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures
    )
    
    plot_model(
        model, 
        to_file=os.path.join(os.path.dirname(fn_args.serving_model_dir), 'model_architecture.png'),
        show_shapes=True,
        show_layer_names=True
    )
