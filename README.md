This project is a workflow to retrain a tensorflow model and convert it to tensorflow lite (quantized or float).

## Generate training set from video
```bash
mkdir -p training_set/mylabel_1

mkdir -p training_set/mylabel_2

docker run -it -v `pwd`:/home jonarod/tflite_tools \
    ffmpeg -i /home/myvideo_1.mp4 /home/training_set/mylabel_1/myvideo_%04d.jpg

docker run -it -v `pwd`:/home jonarod/tflite_tools \
    ffmpeg -i /home/myvideo_2.mp4 /home/training_set/mylabel_2/myvideo_%04d.jpg
```
`myvideo_1.mp4` is a video where you shoot your object under different angles and lighting conditions.
The script will split the video into images and put them into a labeled folder.
`mylabel_1` should be the name you want the model to return when it recognizes your object.

You need at least 2 labels to classify, so you should do it at least twice for 2 or more objects.


## Retrain model
```bash
docker run -it -v `pwd`:/home jonarod/tflite_tools \
    python -m scripts.retrain \
    --bottleneck_dir=/home/my_model/bottlenecks \
    --model_dir=/home/my_model/models/ \
    --summaries_dir=/home/my_model/training_summaries/mobilenet_0.50_224 \
    --output_graph=/home/my_model/retrained_graph.pb \
    --output_labels=/home/my_model/retrained_labels.txt \
    --architecture=mobilenet_0.50_224 \
    --image_dir=/home/training_set
```

## Test model with an image
```bash
docker run -it -v `pwd`:/home jonarod/tflite_tools \
    python -m scripts.label_image \
    --image=/home/test_set/my_random_test_image.jpg \
    --graph=/home/my_model/retrained_graph.pb \
    --labels=/home/my_model/retrained_labels.txt
```


## Convert from FLOAT `.pb` to QUANTIZED `.tflite`
```bash
docker run -it -v `pwd`:/home jonarod/tflite_tools \
    tflite_convert \
    --graph_def_file=/home/my_model/retrained_graph.pb \
    --output_file=/home/my_model/retrained_graph_quant.tflite \
    --output_format=TFLITE \
    --inference_type=QUANTIZED_UINT8 \
    --input_shapes=1,224,224,3 \
    --input_arrays=input \
    --output_arrays=final_result \
    --mean_values=128 \
    --std_dev_values=128 \
    --default_ranges_min=0 \
    --default_ranges_max=100
```

## Convert from FLOAT `.pb` to  FLOAT `.tflite`
```bash
docker run -it -v `pwd`:/home jonarod/tflite_tools \
    tflite_convert \
    --graph_def_file=/home/my_model/retrained_graph.pb \
    --output_file=/home/my_model/retrained_graph_float.tflite \
    --output_format=TFLITE \
    --inference_type=FLOAT \
    --input_shapes=1,224,224,3 \
    --input_arrays=input \
    --output_arrays=final_result
```


## Check tflite model input/output shape and type 
```bash
docker run -it -v `pwd`:/home jonarod/tflite_tools \
    python -m scripts.inspect \
    --tflite_model /home/my_model/retrained_graph_float.tflite
```
`FLOAT` models will output something like:
```bash
[  1 224 224   3]
<class 'numpy.float32'>
[1 3]
<class 'numpy.float32'>
```


```bash
docker run -it -v `pwd`:/home jonarod/tflite_tools \
    python -m scripts.inspect \
    --tflite_model /home/my_model/retrained_graph_quant.tflite
```

`QUANTIZED` models will output something like:
```bash
[  1 224 224   3]
<class 'numpy.uint8'>
[1 3]
<class 'numpy.uint8'>
```
