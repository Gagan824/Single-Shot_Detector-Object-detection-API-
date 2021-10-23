import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
from metrics_functions import mean_average_precision_for_boxes
import json

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-13')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test')
test_path = os.listdir(IMAGE_PATH)
test_images = [image for image in test_path if Path(image).suffix == '.JPG']

products ={}
data =[]
number = 0

print('reading test images for performing validation')
for image in test_images:
    img = cv2.imread(IMAGE_PATH+'/'+image)
    im_height,im_width = img.shape[0],img.shape[1]
    image_np = np.array(img)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    #print(detections)
    count =0
    for i in range(len(detections['detection_boxes'])):
        if detections['detection_scores'][i] > 0.5:
            count=count+1
            xmin,ymin = detections['detection_boxes'][i][1]*im_width,detections['detection_boxes'][i][0]*im_height
            xmax,ymax = detections['detection_boxes'][i][3]*im_width,detections['detection_boxes'][i][2]*im_height
            #print(category_index[detections['detection_classes'][i]+1]['name'],detections['detection_scores'][i],xmin,ymin,xmax,ymax)

            data.append([image,category_index[detections['detection_classes'][i]+1]['name'],detections['detection_scores'][i],\
                        xmin,ymin,xmax,ymax])

    products[image] = count
    number = number+1
        
# detection_classes should be ints.
print(number,'Test Images done')

with open('image2products.json', "w") as f:
    f.write(json.dumps(products,indent=4))
print('Products per shelf has saved to image2products.json')


predicted_data = pd.DataFrame(data,columns= ['Image','class','score','x_min','y_min','x_max','y_max'])

predicted_data.to_csv('data/predicted_data.csv')
print('Predicted data saved to predicted_data.csv in data directory')

ann = pd.read_csv('data/test_data.csv')
det = pd.read_csv('data/predicted_data.csv')
ann = ann[['image', 'class', 'x_min', 'x_max', 'y_min', 'y_max']].values
det = det[['Image', 'class', 'score', 'x_min', 'x_max', 'y_min', 'y_max']].values
mean_ap, average_precisions, r, p = mean_average_precision_for_boxes(ann, det)

metrics = {
            'mAP': mean_ap,
            'precision': p,
            'recall': r[-1]  
            }

print('Validation Metrics of the model:',metrics)

with open('metrics.json', "w") as f:
    f.write(json.dumps(metrics,indent=4))
    
print('Validation Metrics saved with name :metrics.json')    
