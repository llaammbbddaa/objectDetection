# syntax=docker/dockerfile:1

# YOLO11 object detection (C++ / OpenCV DNN)
# Stages:
#   1. opencv-build : builds OpenCV 4.10.0 (core, imgproc, imgcodecs, dnn only)
#   2. app-build    : compiles yolo11_detect against that OpenCV
#   3. model-export : exports yolo11n.pt -> yolo11.onnx (opset=12, simplify=True)
#   4. runtime      : slim image with only the binary, OpenCV libs and the model

ARG UBUNTU_VERSION=24.04
ARG OPENCV_VERSION=4.10.0

# ---------------------------------------------------------------- 1. OpenCV
FROM ubuntu:${UBUNTU_VERSION} AS opencv-build
ARG OPENCV_VERSION
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates pkg-config \
        libjpeg-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git /opencv \
    && cmake -S /opencv -B /opencv/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/opencv \
        -DBUILD_LIST=core,imgproc,imgcodecs,dnn \
        -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
        -DBUILD_opencv_apps=OFF -DBUILD_DOCS=OFF \
    && cmake --build /opencv/build -j"$(nproc)" \
    && cmake --install /opencv/build

# ------------------------------------------------------------ 2. Application
FROM ubuntu:${UBUNTU_VERSION} AS app-build
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake libjpeg-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=opencv-build /opt/opencv /opt/opencv

WORKDIR /src
COPY CMakeLists.txt main.cpp ./
RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DOpenCV_DIR=/opt/opencv/lib/cmake/opencv4 \
    && cmake --build build -j"$(nproc)"

# ---------------------------------------------------------- 3. Model export
FROM python:3.11-slim AS model-export

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps this stage much smaller; it is discarded anyway.
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ultralytics onnx onnxslim

WORKDIR /export
# Downloads yolo11n.pt on first use. opset=12 + simplify=True are required
# for OpenCV's ONNX importer to read the model.
RUN yolo export model=yolo11n.pt format=onnx opset=12 simplify=True \
    && mv yolo11n.onnx yolo11.onnx

# --------------------------------------------------------------- 4. Runtime
FROM ubuntu:${UBUNTU_VERSION} AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg-turbo8 libpng16-16t64 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=opencv-build /opt/opencv/lib /opt/opencv/lib
COPY --from=app-build /src/build/yolo11_detect /usr/local/bin/yolo11_detect
COPY --from=model-export /export/yolo11.onnx /opt/model/yolo11.onnx

ENV LD_LIBRARY_PATH=/opt/opencv/lib

# The program loads "yolo11.onnx" from the current working directory, and
# writes annotated.jpg / detections.csv there. This wrapper links the bundled
# model into whatever directory you mount, unless you supply your own.
RUN printf '%s\n' \
      '#!/bin/sh' \
      '[ -e yolo11.onnx ] || ln -s /opt/model/yolo11.onnx yolo11.onnx' \
      'exec /usr/local/bin/yolo11_detect "$@"' \
      > /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /data
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["input.jpg"]

# Build:  docker build -t yolo11-detect .
# Run:    docker run --rm -v "$PWD":/data yolo11-detect input.jpg 0.25 0.45
#         (results: annotated.jpg and detections.csv appear in your $PWD)
