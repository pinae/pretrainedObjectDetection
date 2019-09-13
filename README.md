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

This project uses TensorFlow 2.0 which is still a release candidate. 
To install it use the following command:

```shell script
pip install setuptools==41.2.0 tensorflow==2.0.0-rc0
```

This also installs a newer version of setuptools.

To access the webcam this project uses OpenCV:

```shell script
pip install opencv-python
```
