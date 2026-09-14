FROM tensorflow/serving:latest

# Sesuaikan dengan base_path di models.config Anda
COPY ./output/serving_model /serving_model 
COPY ./config /model_config

ENV MODEL_NAME=wine-quality-model
ENV MONITORING_CONFIG="/model_config/prometheus.config"

CMD ["sh", "-c", "tensorflow_model_server --port=8500 --rest_api_port=${PORT} --model_config_file=/model_config/models.config --monitoring_config_file=/model_config/prometheus.config"]