#!/usr/bin/env python
import argparse
import tensorflow as tf

# Get args
parser = argparse.ArgumentParser()
parser.add_argument("--tflite_model", help="TFLite model to inspect")
args = parser.parse_args()

# Load model
interpreter = tf.lite.Interpreter(model_path=args.tflite_model)
interpreter.allocate_tensors()

# Print input shape and type
print(interpreter.get_input_details()[0]['shape'])  # Example: [1 299 299 3]
print(interpreter.get_input_details()[0]['dtype'])  # Example: <class 'numpy.float32'>

# Print output shape and type
print(interpreter.get_output_details()[0]['shape'])  # Example: [1 1001]
print(interpreter.get_output_details()[0]['dtype'])  # Example: <class 'numpy.float32'>s
