#!/bin/bash

# Script to build and run the nn_utils tests

echo "=== Building nn_utils library ==="
make build-nn-utils
if [ $? -ne 0 ]; then
    echo "Error: Failed to build nn_utils library"
    exit 1
fi

echo -e "\n=== Building nn_utils test executable ==="
make build-nn-utils-test
if [ $? -ne 0 ]; then
    echo "Error: Failed to build nn_utils test"
    exit 1
fi

echo -e "\n=== Running nn_utils tests ==="
make run-nn-utils-test
if [ $? -ne 0 ]; then
    echo "Error: Tests failed"
    exit 1
fi

echo -e "\n=== All tests passed successfully! ==="
