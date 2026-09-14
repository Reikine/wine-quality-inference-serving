### Components
This is components.py
```py
"""Inisiasi Komponen Pipeline TFX
"""
from dataclasses import dataclass
import os

import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Tuner, 
    Trainer,
    Evaluator,
    Pusher
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

@dataclass
class PipelineConfig:
    """Konfigurasi parameter yang dibutuhkan untuk pipeline TFX."""
    data_dir: str
    transform_module: str
    tuner_module: str
    training_module: str
    training_steps: int
    eval_steps: int
    serving_model_dir: str

def init_components(config: PipelineConfig):
    """Initiate tfx pipeline components using a configuration object.
    
    Args:
        config (PipelineConfig): Objek berisi parameter konfigurasi pipeline.
        
    Returns:
        tuple: Kumpulan komponen TFX
    """
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2),
            ]
        )
    )
    
    example_gen = CsvExampleGen(
        input_base=config.data_dir,
        output_config=output
    )
    
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs['examples']
    )
    
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics']
    )
    
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )
    
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=os.path.abspath(config.transform_module)
    )
    
    tuner = Tuner(
        module_file=os.path.abspath(config.tuner_module),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'],
            num_steps=config.training_steps
        ),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'],
            num_steps=config.eval_steps
        )
    )
    
    trainer  = Trainer(
        module_file=os.path.abspath(config.training_module),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        hyperparameters=tuner.outputs['best_hyperparameters'],
        train_args=trainer_pb2.TrainArgs(
            splits=['train'],
            num_steps=config.training_steps
        ),
        eval_args=trainer_pb2.EvalArgs(
            splits=['eval'], 
            num_steps=config.eval_steps
        )
    )
    
    model_resolver = Resolver(
        strategy_class= LatestBlessedModelStrategy,
        model = Channel(type=Model),
        model_blessing = Channel(type=ModelBlessing)
    ).with_id('Latest_blessed_model_resolver')
    
    slicing_specs = [
        tfma.SlicingSpec()
    ]
    
    metrics_specs = [
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(
                    class_name='SparseCategoricalAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.3}
                        ),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': 0.0001}
                        )
                    )
                )
            ]
        )
    ]
    
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='quality_xf')],
        slicing_specs=slicing_specs,
        metrics_specs=metrics_specs
    )
    
    evaluator = Evaluator(
        examples=transform.outputs['transformed_examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config
    )
    
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=config.serving_model_dir
            )
        )
    )
    
    components = (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher
    )
    
    return components
 
```
___
### Trainer
This is winequality_trainer.py
```py
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

```
___
### Transform
This is winequality_transform.py
```py
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

```
### Pipeline
This is local_pipeline.py
```py
import os
import sys
from typing import Text
 
from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
 
PIPELINE_NAME = 'setyana-dev-pipeline'
 
# pipeline inputs
DATA_ROOT = 'data'
TRANSFORM_MODULE_FILE = 'modules/winequality_transform.py'
TRAINER_MODULE_FILE = 'modules/winequality_trainer.py'
TUNER_MODULE_FILE = 'modules/winequality_trainer.py'
 
# pipeline outputs
OUTPUT_BASE = 'output'
serving_model_dir = os.path.join(OUTPUT_BASE, 'serving_model')
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, 'metadata.sqlite')

def init_local_pipeline(
    components, pipeline_root: Text
) -> pipeline.Pipeline: # type: ignore
    
    logging.info(f'Pipeline root set to: {pipeline_root}')
    beam_args = [
        '--direct_running_mode=in_memory',
        # 0 auto-detect based on on the number of CPUs available 
        # during execution time.
        '--direct_num_workers=1'
    ]
    
    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=beam_args
    )
    
if __name__ == '__main__':
        logging.set_verbosity(logging.INFO)
        
        from modules.components import init_components, PipelineConfig
        
        config = PipelineConfig(
            data_dir=DATA_ROOT,
            transform_module=TRANSFORM_MODULE_FILE,
            tuner_module=TUNER_MODULE_FILE,
            training_module=TRAINER_MODULE_FILE,
            training_steps=40,
            eval_steps=10,
            serving_model_dir=serving_model_dir
        )
        
        components = init_components(config)
        
        pipeline = init_local_pipeline(components, pipeline_root)
        BeamDagRunner().run(pipeline=pipeline)
```