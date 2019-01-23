#!/usr/bin/env python
import tensorflow as tf

graph_def_file = "/home/yoga/Documents/Python/cnn_training/tf_files/retrained_nutella_graph_quant.pb"
input_arrays = ["input"]
output_arrays = ["final_result"]

converter = tf.contrib.lite.TocoConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
converter.inference_type = tf.contrib.lite.constants.QUANTIZED_UINT8
converter.quantized_input_stats = {input_arrays[0] : (0., 1.)}
tflite_model = converter.convert()
open("tf_files/optimized_retrained_nutella_graph_quant.tflite", "wb").write(tflite_model)
