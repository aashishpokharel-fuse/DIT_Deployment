
# The version of cuda must match those of the packages installed in src/Dockerfile.gpulibs<important>
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
ENV DEBIAN_FRONTEND noninteractive

WORKDIR /app

LABEL authors="Aashish Pokharel<aashish.pokharel@fusemachines.com>"
RUN pip install  tensorboard cmake onnx   # cmake from apt-get is too old
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

ENV FORCE_CUDA="1"



RUN apt update && apt-get install sudo wget git  g++ gcc -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip
RUN pip install  'git+https://github.com/facebookresearch/fvcore'
RUN rm -rf build/ **/*.so
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo

RUN pip install -e detectron2_repo

ENV FVCORE_CACHE="/tmp"

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
# RUN pip install shapely

EXPOSE 9000

COPY . /app

ENV MODEL_NAME DocumentStructureModel

ENV SERVICE_TYPE MODEL

CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE
