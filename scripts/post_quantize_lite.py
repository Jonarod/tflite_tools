#!/usr/bin/env python
import tensorflow as tf

# Get args
parser = argparse.ArgumentParser()
parser.add_argument("--graph_def_file", help="Graph model to convert")
parser.add_argument("--input_arrays", help="input arrays")
parser.add_argument("--output_arrays", help="output arrays")
parser.add_argument("--output_file", help="Path to the output TFLite file")
parser.add_argument("--inference_type", help="Inference type [FLOAT|QUANTIZED_UINT8]")
args = parser.parse_args()


# Convert
converter = tf.lite.TocoConverter.from_frozen_graph(args.graph_def_file, args.input_arrays, args.output_arrays)

if args.inference_type == 'QUANTIZED_UINT8':
    converter.post_training_quantize = True
else:
    converter.post_training_quantize = False
    
tflite_model = converter.convert()
open(args.output_file, "wb").write(tflite_model)
