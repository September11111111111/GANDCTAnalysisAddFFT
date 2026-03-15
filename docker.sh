#!/bin/bash
set -e

DATASET_DIRS="/d/datasets/gandct"
IMAGE="dct"

build() {
    docker build -t "$IMAGE" .
}

shell() {
    PROJECT_DIR="$(pwd -W)"
    DATASET_DIR_WIN="$(cygpath -w "$DATASET_DIRS")"

    docker run --rm -it \
      --gpus all \
      -v "$PROJECT_DIR:/dct" \
      -v "$DATASET_DIR_WIN:/dct/datasets" \
      dct bash -lc "cd /dct && exec bash"
}


tests() {
    docker run --rm -it \
      -v "$(pwd):/dct" \
      -w /dct \
      "$IMAGE" pytest
}

clean() {
    rm -rf log ckpt final_models
}

print_usage() {
    echo "Usage: ./docker.sh {build|shell|tests|clean}"
    echo "    build - Build the Docker image."
    echo "    shell - Spawn a shell inside the docker container."
    echo "    tests - Run pytest inside the container."
    echo "    clean - Cleanup directories from training."
}

case "$1" in
  build) build ;;
  shell) shell ;;
  tests) tests ;;
  clean) clean ;;
  ""|*) print_usage; exit 1 ;;
esac
