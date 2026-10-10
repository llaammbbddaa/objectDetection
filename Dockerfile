# syntax=docker/dockerfile:1

# YOLO26 object detection (C++ / OpenCV DNN)
# Stages:
#   1. opencv-build : builds OpenCV 4.11.0 (core, imgproc, imgcodecs, dnn only)
#   2. app-build    : compiles yolo26_detect against that OpenCV
#   3. model-export : exports yolo26n.pt -> yolo26.onnx
#   4. runtime-base : slim image with only the OpenCV libs
#   5. smoke-test   : loads the exported model with OpenCV and runs one forward
#                     pass, so an incompatible model fails the BUILD, not your run
#   6. runtime      : runtime-base + binary + model (the final image)

ARG UBUNTU_VERSION=24.04
ARG OPENCV_VERSION=4.11.0

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

# Export settings. The defaults give YOLO26's end-to-end (NMS-free) model,
# output [1, 300, 6]. If OpenCV can't run that graph (the smoke test below
# will fail), rebuild with:  --build-arg END2END=False
# which exports the classic [1, 84, 8400] layout; main.cpp handles both.
ARG END2END=True
ARG OPSET=12

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps this stage much smaller; it is discarded anyway.
# YOLO26 needs a recent Ultralytics release, so ultralytics is left unpinned.
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ultralytics onnx onnxslim

WORKDIR /export
# Downloads yolo26n.pt on first use.
RUN yolo export model=yolo26n.pt format=onnx opset=${OPSET} simplify=True end2end=${END2END} \
    && mv yolo26n.onnx yolo26.onnx

# ------------------------------------------------------------ 4. Runtime base
FROM ubuntu:${UBUNTU_VERSION} AS runtime-base

RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg-turbo8 libpng16-16t64 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=opencv-build /opt/opencv/lib /opt/opencv/lib
ENV LD_LIBRARY_PATH=/opt/opencv/lib

# ------------------------------------------------------------- 5. Smoke test
FROM runtime-base AS smoke-test

COPY --from=app-build /src/build/yolo26_detect /usr/local/bin/yolo26_detect
COPY --from=model-export /export/yolo26.onnx /smoke/yolo26.onnx

# A blank 640x480 PPM (OpenCV reads PPM natively). Zero detections is fine;
# what matters is that the model loads and a forward pass completes.
WORKDIR /smoke
RUN { printf 'P6\n640 480\n255\n'; head -c 921600 /dev/zero; } > blank.ppm \
    && yolo26_detect blank.ppm \
    && touch /smoke/ok

# --------------------------------------------------------------- 6. Runtime
FROM runtime-base AS runtime

COPY --from=app-build /src/build/yolo26_detect /usr/local/bin/yolo26_detect
COPY --from=model-export /export/yolo26.onnx /opt/model/yolo26.onnx
# Forces the smoke test to run on every build (BuildKit skips unused stages).
COPY --from=smoke-test /smoke/ok /opt/model/.smoke-test-passed

# The program loads "yolo26.onnx" from the current working directory, and
# writes annotated.jpg / detections.csv there. This wrapper links the bundled
# model into whatever directory you mount, unless you supply your own.
RUN printf '%s\n' \
      '#!/bin/sh' \
      '[ -e yolo26.onnx ] || ln -s /opt/model/yolo26.onnx yolo26.onnx' \
      'exec /usr/local/bin/yolo26_detect "$@"' \
      > /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /data
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["input.jpg"]

# Build:  docker build -t yolo26-detect .
# Run:    docker run --rm -v "$PWD":/data yolo26-detect input.jpg 0.25 0.45
#         (results: annotated.jpg and detections.csv appear in your $PWD)
