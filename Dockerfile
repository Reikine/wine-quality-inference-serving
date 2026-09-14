FROM tensorflow/serving:latest

COPY ./output/serving_model /models/wine-quality-model

ENV MODEL_NAME=wine-quality-model

CMD ["sh", "-c", "tensorflow_model_server --port=8500 --rest_api_port=${PORT} --rest_api_address=0.0.0.0 --model_name=${MODEL_NAME} --model_base_path=/models/${MODEL_NAME}"]
