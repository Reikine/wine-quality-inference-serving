# PIPELINE MLOPS WINE QUALITY PREDICTION
Dalam pengembangan sistem MLOps untuk Wine Quality Prediction dengan pendekatan secara end-to-end machine learning, digunakan dua framework utama yaitu TensorFlow dan TensorFlow Extended (TFX). TensorFlow digunakan untuk membangun, melatih, dan mengevaluasi arsitektur model deep learning yang akan memprediksi kualitas wine. Sementara itu, TensorFlow Extended (TFX) digunakan sebagai platform produksi untuk mengotomatisasi seluruh siklus data dan model mulai dari data ingestion, validasi skema, transformasi fitur, hingga manajemen perilisan model ke tahap produksi secara konsisten dan terukur.
```py
import tensorflow as tf
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Tuner
from tfx.proto import example_gen_pb2
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
import os
```
___
## Pipeline Environment Setup & Variable Initialization
Tahap ini akan menginisialisasi dan mengkonfigurasi jalur (path) serta variabel lingkungan (environment variables) yang dibutuhkan oleh **TensorFlow Extended (TFX)** dalam mengelola seluruh *artifact* dan *metadata* pipeline:
* `PIPELINE_NAME` & `SCHEMA_PIPELINE_NAME`: Membuat identitas unik untuk pipeline `setyana-dev-pipeline` dan skema data.
* `PIPELINE_ROOT`: Menetapkan direktori penyimpanan seluruh output *artifact* yang berisi data `TFRecord`, grafik `Transform`, checkpoint `Trainer`, dan hasil evaluasi `TFMA`.
* `METADATA_PATH`: Menetapkan lokasi SQLite (`metadata.db`) untuk ML Metadata (MLMD) yang berisi *data lineage*, konfigurasi parameter, dan *execution history* dari pipeline.
* `SERVING_MODEL_DIR`: Menetapkan direktori untuk menyimpan *SavedModel* terbaik yang nantinya diekspor oleh `Pusher` dan di-mount ke TensorFlow Serving atau Docker.
* `DATA_ROOT`: Menetapkan direktori dataset `winequality-red.csv`.
* `InteractiveContext`: Inisialisasi *orchestrator* yang berfungsi dalam  mengontrol alur eksekusi komponen TFX satu per satu serta mencatat *metadata* eksekusi ke MLMD.
```py
PIPELINE_NAME = "setyana-dev-pipeline"
SCHEMA_PIPELINE_NAME = "winequality-red-tfdv-schema"
PIPELINE_ROOT = os.path.join('pipelines', PIPELINE_NAME)
METADATA_PATH = os.path.join('metadata', PIPELINE_NAME, 'metadata.db')
SERVING_MODEL_DIR = os.path.join('serving_model', PIPELINE_NAME)
DATA_ROOT = "data"
interactive_context = InteractiveContext(pipeline_root=PIPELINE_ROOT)
```
___
## Data Ingestion
Data ingestion adalah proses mengumpulkan, mengambil, dan memasukkan data mentah agar siap diolah lebih lanjut untuk keperluan keperluan pengembangan *MLOps*. Untuk itu digunakanlah komponen `CsvExampleGen` karena data `winequality-red` memiliki format ekstensi csv dan menunjuk `DATA_ROOT` sebagai tempat penyimpanan data mentah. Komponen `CsvExampleGen` akan mengonversi data yang masuk menjadi format standar `TFRecord (tf.train.Example)`, dan membagi data menjadi data latihan (train) dan data evaluasi (eval).
```py
# ExampleGen
output = example_gen_pb2.Output(
    split_config = example_gen_pb2.SplitConfig(
        splits=[
            example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2)
        ]
    )
)

example_gen = CsvExampleGen(input_base=DATA_ROOT, output_config=output)

interactive_context.run(example_gen)
```
___
## Data Validation
Data validation merupakan komponen yang memeriksa kualitas dan perubahan dari suatu data dengan memeriksa data baru dan membandingkannya dengan dataset yang digunakan dalam proses training. Proses data validation dapat dilakukan dengan beberapa cara, salah satunya adalah menggunakan data schema yang memerlukan tahapan:
* Pemeriksaan parameter statistik 
* Pemeriksaan data schema
* Pemeriksaan anomali

### Pemeriksaan Parameter Statistik
Tahap ini akan menganalisis dataset untuk menghitung statistik deskriptif dari setiap fitur fisikokimia (seperti mean, standar deviasi, nilai min/max, dan nilai yang hilang). Proses ini akan menggunakan komponen pada TFX yang bernama `StatisticsGen` dan hasil outputnya akan disimpan sebagai `statisttic`. 
```py
# StatisticGen
statistic_gen = StatisticsGen(
    examples=example_gen.outputs["examples"]
)

interactive_context.run(statistic_gen)
```
Untuk menampilkan visualisasi interaktif hasil kalkulasi statistik, dapat menggunakan perintah `interactive_context.show(statistic_gen.outputs["statistics"])` yang akan menampilkannya dalam bentuk visual TFDV.
```py
interactive_context.show(statistic_gen.outputs["statistics"])
```
Tahap data schema menghasilkan Skema Data (Schema), menggunakan komponen TFX bernama `SchemaGen` dengan menggunakan hasil dari `statistic` sebelumnya. Data schema ini berisi nama fitur, tipe data, serta batasan nilai yang valid.
```py
# SchemaGen
schema_gen = SchemaGen(
    statistics=statistic_gen.outputs["statistics"]
)

interactive_context.run(schema_gen)
interactive_context.show(schema_gen.outputs["schema"])
```
Komponen berikutnya adalah `ExampleValidator` yang digunakan untuk memvalidasi dataset terhadap skema yang telah dibuat guna mendeteksi adanya anomali atau data drift yang dapat mengganggu performa model. Komponen ini menerima input berupa `statistics` dan `schema`.
```py
# ExampleValidator
example_validator = ExampleValidator(
    statistics=statistic_gen.outputs["statistics"],
    schema=schema_gen.outputs["schema"]
)

interactive_context.run(example_validator)
interactive_context.show(example_validator.outputs["anomalies"])
```
___
### Data Preprocessing
Data Preprocessing merupakan sebuah proses untuk mengubah (*transform*) suatu data ke dalam format yang lebih mudah diproses oleh mesin dengan manipulasi fitur seperti scaling, one-hot encoding, imputation, atau pembuatan engineered features.

Mendefinisikan nama modul Python bernama `winequality_transform.py` agar kodenya modular dan mudah dipanggil ulang.
```py
TRANSFORM_MODULE_FILE = "winequality_transform.py"
```
Membuat modul yang berisi logika transformasi data, seperti scaling untuk fitur numerik dan pengubahan tipe data (casting) untuk variabel target.
```py
%%writefile {TRANSFORM_MODULE_FILE}
"""Modul preprocessing data untuk pipeline TFX wine quality."""

import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = "quality"
NUMERIC_FEATURE_KEYS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

def transformed_name(key):
    """Menambhakan akhiran _xf pada nama fitur."""
    return key + "_xf"

def preprocessing_fn(inputs):
    """Melakukan scaling scale_to_z_score pada fitur numerik dan konversi tipe data label.'"""
    outputs = {}
    
    for key in NUMERIC_FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.scale_to_z_score(inputs[key])
        
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY] - 3, tf.int64)
    
    return outputs

```
Menjalankan komponen `Transform` dari TFX yang menerima input berupa data `(examples)`, skema `(schema)`, dan file modul `(module_file)` untuk menerapkan transformasi ke pipeline. Komponen ini menghasilkan `transform graph` yang akan digunakan pada saat training maupun serving, sehingga mencegah terjadinya data leakage dan inkonsistensi preprocessing.
```py
# Transform
transform = Transform(
    examples=example_gen.outputs["examples"],
    schema=schema_gen.outputs["schema"],
    module_file=os.path.abspath(TRANSFORM_MODULE_FILE)
)

interactive_context.run(transform)
```
___
## Trainer
Trainer merupakan komponen dalam pipeline machine learning yang bertugas untuk melakukan pelatihan model secara otomatis menggunakan data yang telah diproses oleh `Transform`, dengan menerapkan arsitektur jaringan saraf serta hyperparameter optimal guna menghasilkan model siap pakai `(SavedModel)`.

Mendefinisikan nama berkas modul `winequality_trainer.py` yang akan menyimpan logika arsitektur model, hyperparameter tuning, serta fungsi pelatihan (training) agar kodenya modular.
```py
TRAINER_MODULE_FILE = "winequality_trainer.py"
```
```py
%%writefile {TRAINER_MODULE_FILE}
"""Modul trainer untuk pipeline TFX wine quality."""

import tensorflow as tf
import keras_tuner as kt
import os
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult

LABEL_KEY = 'quality'

def transformed_name(key):
    """"Mengubah nama fitur yang telah melalui proses transform."""
    return key + "_xf"

def _gzip_reader_fn(filenames):
    """Memuat data dalam format TFRecord."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def _input_fn(file_pattern, tf_transform_output, batch_size=32):
    """"Membaca data hasil transformasi dari komponen Transform dan mengembalikannya sebagai dataset berformat batch."""
    
    # Mengambil spesifikasi fitur hasil transformasi dari output Transform.
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )
    
    # Membuat batch data.
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY)
    )
    
    return dataset
    
def model_builder(hp, transform_output):
    """Membangun dan mengompilasi model Keras dengan hyperparameter yang dinamis."""
    
    # Mendefinisikan ruang lingkup pencarian hyperparameter.
    hp_units_1 = hp.Int('units_1', min_value=32, max_value=128, step=32)
    hp_units_2 = hp.Int('units_2', min_value=16, max_value=64, step=16)
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3])
    
    # Mengambil spesifikasi fitur dan menghapus label target.
    feature_spec = transform_output.transformed_feature_spec().copy()
    feature_spec.pop(transformed_name(LABEL_KEY))
    
    # Membuat input layer untuk setiap fitur.
    inputs = {
        key: tf.keras.Input(
            shape=(1,),
            name=key,
        ) for key in feature_spec.keys()
    }
    
    # Membuat arsitektur Neural Network.    
    x = layers.Concatenate(axis=-1)(list(inputs.values()))
    x = layers.Dense(hp_units_1, activation='relu')(x)
    x = layers.Dense(hp_units_2, activation='relu')(x)
    outputs=layers.Dense(6, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Mengkompilasi model dengan optimasi dan loss function.
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
        metrics=['sparse_categorical_accuracy']
    )
    
    model.summary()
    
    return model

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Membangun fungsi inferensi model untuk menangani data mentah saat tahap produksi."""
    
    # Mengintegrasikan layer preprocessing TFT ke dalam arsitektur model.
    model.tft_layer = tf_transform_output.transform_features_layer()
    
    @tf.function
    def serve_tf_example_fn(serialized_tf_examples):
        # Mengambil spesifikasi fitur mentah dan menghapus label target.
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        
        # Memuat (parsing) data mentah dan mengaplikasikan grafik transformasi
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        
        # Mengembalikan hasil prediksi dari model.
        return model(transformed_features)
    
    return serve_tf_example_fn

def tuner_fn(fn_args: FnArgs):
    """Mengeksekusi proses pencarian hyperparameter terbaik untuk model."""
    
    # Memuat output dari komponen Transform.
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    # Memuat dataset latih dan evaluasi dalam bentuk batch.
    train_set = _input_fn(fn_args.train_files, tf_transform_output, batch_size=32)
    val_set = _input_fn(fn_args.eval_files, tf_transform_output, batch_size=32)
    
    # Inisialisasi tuner menggunakan metode RandomSearch.
    tuner = kt.RandomSearch(
        hypermodel=lambda hp: model_builder(hp, tf_transform_output),
        objective='val_sparse_categorical_accuracy',
        max_trials=5,
        directory=fn_args.working_dir,
        project_name='wine_quality_tuning'
    )
    
    # Mengembalikan hasil konfigurasi tuner dan argumen eksekusi pelatihan.
    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_set,
            'validation_data': val_set,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )


def run_fn(fn_args: FnArgs) -> None:
    """Mengeksekusi proses pelatihan model menggunakan hyperparameter terbaik."""
    
    # Menentukan direktori untuk pencatatan TensorBoard.
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'log')
    
    # Mendefinisikan callback TensorBoard dan EarlyStopping.
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq='batch'
    )
    
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_sparse_categorical_accuracy',
        mode='max',
        verbose=1,
        patience=10
    )
    
    # Memuat output dari komponen Transform.
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    # Memuat dataset latih dan evaluasi dalam bentuk batch.
    train_set = _input_fn(fn_args.train_files, tf_transform_output)
    val_set = _input_fn(fn_args.eval_files, tf_transform_output)
    
    # Memuat hyperparameter terbaik dari tuner jika tersedia.
    if fn_args.hyperparameters:
        hp = kt.HyperParameters.from_config(fn_args.hyperparameters)
    else:
        hp = kt.Hyperparameters()
    
    # Membangun arsitektur model berdasarkan hyperparameter.
    model = model_builder(hp, tf_transform_output)
    
    # Melatih model dengan dataset dan callback yang telah dikonfigurasi.
    model.fit(
        x=train_set,
        validation_data=val_set,
        callbacks=[tensorboard_callback, es],
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        epochs=10
    )
    
    # Menentukan serving signature untuk proses inferensi data mentah.
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
    
    # Menyimpan model yang telah dilatih ke dalam direktori tujuan.
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)

```
Komponen Tuner menggunakan library `keras_tuner` melalui algoritma `RandomSearch` untuk mencari kombinasi hyperparameter paling optimal secara otomatis. Pencarian dilakukan pada rentang unit untuk dua hidden layer serta nilai learning rate. Hasil kombinasi terbaik `(best hyperparameters)` disimpan sebagai `artifact` yang nantinya akan digunakan secara otomatis oleh komponen Trainer.
```py
# Tuner
from tfx.components import Tuner
from tfx.proto import trainer_pb2

tuner = Tuner(
    module_file=TRAINER_MODULE_FILE,
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(splits=['train'], num_steps=20),
    eval_args=trainer_pb2.EvalArgs(splits=['eval'], num_steps=10)
)

interactive_context.run(tuner)
```
Komponen Trainer melatih model Deep Neural Network (DNN) menggunakan modul `winequality_trainer.py`. Modul ini terdiri dari beberapa fungsi utama yang saling terintegrasi:
* `_input_fn`: Membaca berkas `TFRecord` hasil pemrosesan `Transform` dan mengonversinya menjadi objek `tf.data.Dataset` yang siap digunakan untuk pelatihan model dalam skala batch.

* `model_builder`: Membangun arsitektur model DNN dengan parameter terbaik dari komponen Tuner. Model menggunakan dua hidden layer beraktivasi `ReLU`, output layer Dense(10) beraktivasi `Softmax` untuk klasifikasi `multiclass`, serta dioptimasi menggunakan `Adam Optimizer` dan `loss SparseCategoricalCrossentropy`.

* `_get_serve_tf_examples_fn`: Mendefinisikan serving signature agar model yang diekspor dapat menerima sampel input raw string `(tf.train.Example)` dan menjalankan transformasi fitur internal secara otomatis saat proses inferensi.

* `run_fn`: Mengatur alur eksekusi pelatihan secara penuh, mencatat `metric logs` ke `TensorBoard`, menerapkan callback `EarlyStopping` untuk mencegah `overfitting`, serta menyajikan `SavedModel` siap pakai.
```py
# Trainer
from tfx.proto import trainer_pb2

trainer = Trainer(
    module_file=TRAINER_MODULE_FILE,
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    hyperparameters=tuner.outputs['best_hyperparameters'],
    train_args=trainer_pb2.TrainArgs(splits=['train'], num_steps=20),
    eval_args=trainer_pb2.EvalArgs(splits=['eval'], num_steps=10)
)

interactive_context.run(trainer)
```
## Model Analysis and Validation
Tahap ini bertujuan untuk mengevaluasi performa model secara menyeluruh dan menguji kelayakannya sebelum disebarkan ke lingkungan produksi. Proses ini memastikan model yang dihasilkan memenuhi standar kualitas serta aman untuk digunakan pada tahap deployment, komponen ini terbagi menjadi:
* Resolver
* Evaluator
* Pusher

Komponen `Resolver` akan mengambil model terbaik dari iterasi sebelumnya `(baseline model)` untuk dijadikan pembanding.
```py
# Resolver
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

model_resolver = Resolver(
    strategy_class=LatestBlessedModelStrategy,
    model=Channel(type=Model),
    model_blessing=Channel(type=ModelBlessing)
).with_id('Latest_blessed_model_resolver')

interactive_context.run(model_resolver)
```
Komponen `Evaluator` bertugas mengevaluasi performa model secara mendalam menggunakan TFMA (TensorFlow Model Analysis). Komponen ini membandingkan metrik `SparseCategoricalAccuracy` model baru terhadap kriteria threshold yang telah ditentukan yaitu `value_threshold minimal 0.3` dan `change_threshold minimal 0.0001` dibanding `model_baseline`. Jika model memenuhi semua syarat, model akan diberi label `blessed`.
```py
# Evaluator 
import tensorflow_model_analysis as tfma
from tfx.components import Evaluator

eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='quality_xf')],
    slicing_specs=[tfma.SlicingSpec()],
    metrics_specs=[
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(
                    class_name='SparseCategoricalAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(lower_bound={'value': 0.3}),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': 0.0001}
                        )
                    )
                )
            ]
        )
    ]
)

evaluator = Evaluator(
    examples=transform.outputs['transformed_examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config
)

interactive_context.run(evaluator)
```
Pada tahap ini, hasil evaluasi dari komponen `Evaluator` dimuat kembali menggunakan TensorFlow Model Analysis (TFMA) untuk me-render metrik performa secara visual maupun tekstual. Proses ini bertujuan untuk memverifikasi secara mendalam nilai metrik `SparseCategoricalAccuracy` serta mengecek apakah model berhasil memenuhi kriteria *threshold* hingga mendapatkan status `blessed`.
```py
# Visualize the evaluation results
eval_result = evaluator.outputs['evaluation'].get()[0].uri
tfma_result = tfma.load_eval_result(eval_result)
tfma.view.render_slicing_metrics(tfma_result)
tfma.addons.fairness.view.widget_view.render_fairness_indicator(
    tfma_result
)

```
Komponen `Pusher` akan memeriksa status validasi dari komponen `Evaluator`. Jika model mendapatkan status `blessed`, `Pusher` secara otomatis mengekspor berkas model akhir `(SavedModel)` ke direktori serving tujuan. Model di direktori serving ini yang kemudian di-mount dan dijalankan dalam kontainer Docker menggunakan TensorFlow Serving untuk siap melayani `prediction request` via REST API.
```py
# Pusher
from tfx.components import Pusher
from tfx.proto import pusher_pb2

pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory='serving_model/wine-quality-model'
        )
    )
)

interactive_context.run(pusher)
```
___
## Testing
```py
import base64
import requests
import tensorflow as tf

sample_example = tf.train.Example(
    features=tf.train.Features(
        feature={
            'fixed acidity': tf.train.Feature(float_list=tf.train.FloatList(value=[12.0])), # Kunci: Sangat asam
            'volatile acidity': tf.train.Feature(float_list=tf.train.FloatList(value=[1.20])),
            'citric acid': tf.train.Feature(float_list=tf.train.FloatList(value=[0.00])),
            'residual sugar': tf.train.Feature(float_list=tf.train.FloatList(value=[1.2])),
            'chlorides': tf.train.Feature(float_list=tf.train.FloatList(value=[0.20])),
            'free sulfur dioxide': tf.train.Feature(float_list=tf.train.FloatList(value=[5.0])),
            'total sulfur dioxide': tf.train.Feature(float_list=tf.train.FloatList(value=[15.0])),
            'density': tf.train.Feature(float_list=tf.train.FloatList(value=[1.0020])),
            'pH': tf.train.Feature(float_list=tf.train.FloatList(value=[3.80])),
            'sulphates': tf.train.Feature(float_list=tf.train.FloatList(value=[0.30])),
            'alcohol': tf.train.Feature(float_list=tf.train.FloatList(value=[8.0])) # Kunci: Alkohol sangat rendah
        }
    )
)

serialized_example = sample_example.SerializeToString()
b64_example = base64.b64encode(serialized_example).decode('utf-8')

endpoint = 'http://localhost:8501/v1/models/wine-quality-model:predict'
payload = {
    'instances': [
        {'b64': b64_example}
    ]
}

response = requests.post(endpoint, json=payload)

print('Status:', response.status_code)
print('Prediction Response:', response.json())
```