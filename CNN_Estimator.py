from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

