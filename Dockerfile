FROM pytorch/torchserve:0.11.1-cpp-dev-gpu AS base

WORKDIR /home/model-server

USER root

# install dependencies
RUN python3 -m pip install --upgrade pip
RUN pip3 install transformers[torch]

FROM base AS serve-gcs

WORKDIR /home/model-server

# copy model artifacts, custom handler and other dependencies
COPY ./predictor/custom_handler.py ./
COPY ./predictor/index_to_name.json ./
COPY ./predictor/pytorch-cloud-example/ ./
COPY ./predictor/config.properties ./


# expose health and prediction listener ports from the image
EXPOSE 8080
EXPOSE 8081

RUN mkdir -p /home/model-server/model-store

# create model archive file packaging model artifacts and dependencies
RUN torch-model-archiver --force --model-name finetuned-bert-classifier --serialized-file model.safetensors --version 1.0 --handler custom_handler.py --export-path /home/model-server/model-store --extra-files "/home/model-server/config.json,/home/model-server/tokenizer.json,/home/model-server/training_args.bin,/home/model-server/tokenizer_config.json,/home/model-server/special_tokens_map.json,/home/model-server/vocab.txt,/home/model-server/index_to_name.json"


# run Torchserve HTTP serve to respond to prediction requests
CMD ["torchserve", \
    "--start", \
    "--ts-config=/home/model-server/config.properties", \
    "--models", \
    "finetuned-bert-classifier=finetuned-bert-classifier.mar", \
    "--model-store", \
    "/home/model-server/model-store", \
    "--foreground"]
