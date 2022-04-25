#!/bin/bash

cd .data
echo "Downloading data"
gdown "https://drive.google.com/uc?id=1YoNA_ZcFTOYufWosCw2s7QgWMHcjc4Ko"

echo "Extracting data"
tar -xzvf box-mlc-iclr-2022-data.tar.gz
rm box-mlc-iclr-2022-data.tar.gz

cd ..

echo "Successfully downloaded the data"