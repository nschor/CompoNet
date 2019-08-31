# CompoNet: Learning to Generate the Unseen by Part Synthesis and Composition
Created by <a href="https://www.linkedin.com/in/schor-nadav/" target="_blank">Nadav Schor</a>, <a href="https://orenkatzir.github.io/" target="_blank">Oren Katzir</a>, <a href="https://www.cs.sfu.ca/~haoz/" target="_blank">Hao Zhang</a>, <a href="https://www.cs.tau.ac.il/~dcor/" target="_blank">Daniel Cohen-Or</a>.

![representative](https://github.com/nschor/CompoNet/blob/master/images/network_architecture.png)


## Introduction
This work is based on our [ICCV paper](https://arxiv.org/abs/1811.07441). We present CompoNet, a generative neural network for 3D shapes that is based on a part-based prior, where the key idea is for the network to synthesize shapes by varying both the shape parts and their compositions.


## Citation
If you find our work useful in your research, please consider citing:

	@article{schor2018componet,
	  title={CompoNet: Learning to Generate the Unseen by Part Synthesis and Composition},
	  author={Schor, Nadav and Katzir, Oren and Zhang, Hao and Cohen-Or, Daniel},
	  journal={arXiv preprint arXiv:1811.07441},
	  year={2018}
	}


## Dependencies
Requirements:
- Python 2.7
- Tensorflow (version 1.4+)
- OpenCV (for visualization)

Our code has been tested with Python 2.7, TensorFlow 1.4.0, CUDA 8.0 and cuDNN 6.0 on Ubuntu 18.04.


## Installation
Download the source code from the git repository:
```
git clone https://github.com/nschor/CompoNet
```
Compile the Chamfer loss file, under `CompoNet/tf_ops/nn_distance`, taken from [Fan et. al](https://github.com/fanhqme/PointSetGeneration).
```
cd CompoNet/tf_ops/nn_distance
```

Modify the Tensorflow and CUDA path in the `tf_nndistance_compile.sh` script and run it.
```
sh tf_nndistance_compile.sh
```
For visualization go to `utils/`.
```
cd CompoNet/utils
```
Run the `compile_render_balls_so.sh` script.
```
sh compile_render_balls_so.sh
```

If you are using Anaconda, we attached the environment we used `CompoNet.yml` under `anaconda_env/`.
Create the environment using:
```
cd CompoNet/anaconda_env
conda env create -f CompoNet.yml
```
Activate the environment:
```
source activate CompoNet
```
### Data Set
Download the ShapeNetPart dataset by running the `download_data.sh` script under `datasets/`.
```
cd CompoNet/datasets
sh download_data.sh
```
The point-clouds will be stored in `CompoNet/datasets/shapenetcore_partanno_segmentation_benchmark_v0`


### Train CompoNet
To train CompoNet on the Chair category with 400 points per part run:
```
python train.py
```
Check the available options using:
```
python train.py -h
```

### Generate Shapes Using the Trained Model
To generate new shapes, and visualize them run:
```
python test.py --category category --model_path model_path
```
Check the available options using:
```
python test.py -h
```

## License
This project is licensed under the terms of the MIT license (see LICENSE for details).
