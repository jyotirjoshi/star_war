import os
import numpy as np
import random
import tensorflow as tf

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Disable eager execution which can help with some freezing issues
tf.compat.v1.disable_eager_execution()

# Reduce TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Applied CPU-only and performance fixes")
print("Run your training command again!")