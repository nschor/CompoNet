#!/bin/bash
wget https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0.zip
unzip shapenetcore_partanno_segmentation_benchmark_v0.zip
rm shapenetcore_partanno_segmentation_benchmark_v0.zip
mkdir -p data
mv shapenetcore_partanno_segmentation_benchmark_v0 data