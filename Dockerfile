FROM python:3.6-slim

RUN pip install tf_nightly
RUN apt-get update && apt-get -y install ffmpeg && rm -r /var/lib/apt/lists

ADD scripts/ /usr/local/bin/scripts

RUN chmod +x /usr/local/bin/scripts/convert_quantized_tflite.py
RUN chmod +x /usr/local/bin/scripts/count_ops.py
RUN chmod +x /usr/local/bin/scripts/dump_operations.py
RUN chmod +x /usr/local/bin/scripts/evaluate.py
RUN chmod +x /usr/local/bin/scripts/graph_pb2tb.py
RUN chmod +x /usr/local/bin/scripts/hello.py
RUN chmod +x /usr/local/bin/scripts/inspect.py
RUN chmod +x /usr/local/bin/scripts/label_image.py
RUN chmod +x /usr/local/bin/scripts/post_quantize_lite.py
RUN chmod +x /usr/local/bin/scripts/quantize_graph.py
RUN chmod +x /usr/local/bin/scripts/retrain.py
RUN chmod +x /usr/local/bin/scripts/show_image.py
RUN chmod +x /usr/local/bin/scripts/__init__.py
RUN chmod +x /usr/local/bin/scripts/__init__.pyc
RUN chmod +x /usr/local/bin/scripts/retrainV2.py

WORKDIR /usr/local/bin
