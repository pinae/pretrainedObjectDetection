# Pretrained Object Detection
Detect objects in front of the webcam with a pretrained network.

## Installation

I recommend creating a virtualenv for this project to prevent 
packages used in this project to interfere with those installed 
on your base system. A virtualenv is created and activated with 
the following commands:

```shell script
python3 -m venv env
source env/bin/activate
```

At least on Ubuntu 18.04 which comes with a pretty old version of 
`pip` you want to update it first and also install `wheel` to get 
rid of errors when `pip` needs to compile packages:

```shell script
pip install -U pip wheel
```

This project uses TensorFlow 2.0. To date TensorFlow is only 
compatible to Python up to Version 3.7. Consider this if your
sytem Python is already 3.8.
  
To install Tensorflow use the following command:

```shell script
pip install setuptools tensorflow
```

This also installs a newer version of setuptools.

To access the webcam this project uses OpenCV:

```shell script
pip install opencv-python
```

## Tests and preparation

Test the Webcam with:

```shell script
python capture.py
```

Test your ILSVRC2012 Dataset installation. It is expected to live 
in `../../Datasets/ILSVRC2012/` (images in the subfolder `images/`):

```shell script
python ilsvrc2012_dataset.py
```

Make sure you have a JSON version of the labels. You can create the file 
from the `meta.mat` file in the ImageNet Dataset distribution with the
following script:

```shell script
python convert_ilsvrc2012_labels.py
```

## Running

Test the network as Keras Model with fp32:

```shell script
python estimator.py
```

Convert it to a tfLite Model:

```shell script
python convert_to_tflite.py
```

Run the big network with tfLite instead of Keras:

```shell script
python tflite_estimator.py
```

Now you can quantize it to int8:

```shell script
python quantize.py
```

Run the quantized network with images from your webcam:

```shell script
python quantized_estimator.py
```

Benchmark the big network (fp32) versus the quantized one (int8):

```shell script
python benchmark.py
```