# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_hub as hub

tf.compat.v1.disable_eager_execution()
detector = hub.Module("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1")
