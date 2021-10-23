# importing mendatory libraries
from glob import glob
import numpy as np
from tqdm .auto import tqdm
import warnings, os
import matplotlib.pyplot as plt
from PIL import Image
warnings.filterwarnings("ignore")
import shutil
import wget
import pandas as pd
from lxml import etree
import pandas as pd
import xml.etree.cElementTree as ET



# function to read file
def rea(d):
    with open(d,"r") as f:
        return f.read()
    
if os.path.exists('data/'):
    pass
    
else:    
    # creating necessary directories
    os.mkdir(f"data/")
    os.mkdir(f"data/dataset")
    os.mkdir(f"data/dataset/Annotations")
    os.mkdir(f"data/dataset/JPEGImages")
    os.mkdir(f"data/dataset/ImageSets")
    os.mkdir(f"data/dataset/ImageSets/Main")
    print('data directory created')
    # defining the paths
    # path to the annotation .csv file
    path = 'data/grocerydataset/annotations.csv'
    # path to save all the .xml files
    annot_save = 'data/dataset/Annotations/'
    # path to save all the images to train and test
    image_save = 'data/dataset/JPEGImages/'
    # path to train and test splits.
    split_save = 'data/dataset/ImageSets/Main/'
    # path to the training image folder
    train_ = 'data/grocerydataset/ShelfImages/train/'
    # path to the testing image flder
    test_= 'data/grocerydataset/ShelfImages/test/'
    print('path created')
    
    # downloading and moving data to working folders
    os.system(f"git clone https://github.com/gulvarol/grocerydataset.git")
    #os.system(f"wget wget https://storage.googleapis.com/open_source_datasets/ShelfImages.tar.gz")
    wget.download('https://storage.googleapis.com/open_source_datasets/ShelfImages.tar.gz')
    os.system(f"tar -xvf ShelfImages.tar.gz")
    shutil.move(f"ShelfImages/" , "grocerydataset/")
    shutil.move(f"grocerydataset/" , "data/")
    print('data moved')

annotations = rea(path).split('\n')


data = {}
for i in tqdm(annotations):
    d = i.split(',')
    if not d[0]:
        continue
    try:w,h = Image.open(train_+d[0]).size
    except: w,h = Image.open(test_+d[0]).size
    if d[0] in data.keys():
        data[d[0]].append(d[1:]+[h,w])
    else: 
        data[d[0]] = [d[1:] + [h,w]]
        
        
def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
            

for k,v in tqdm(data.items()):
    height = v[0][-2]
    width = v[0][-1]
    depth = 3
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = 'images'
    ET.SubElement(annotation, 'filename').text = str(k)
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)
    ET.SubElement(annotation, 'segmented').text = '0'

    for j in v:
        ob = ET.SubElement(annotation, 'object')
        ET.SubElement(ob, 'name').text = "b" # str(j[4]) class of the bbox
        ET.SubElement(ob, 'pose').text = 'Unspecified'
        ET.SubElement(ob, 'truncated').text = '0'
        ET.SubElement(ob, 'difficult').text = '0'
        bbox = ET.SubElement(ob, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(j[0])
        ET.SubElement(bbox, 'ymin').text = str(j[1])
        ET.SubElement(bbox, 'xmax').text = str(j[2])
        ET.SubElement(bbox, 'ymax').text = str(j[3])
    
    fileName = str(k.split('.')[0])
    tree = ET.ElementTree(annotation)
    indent(annotation)
    tree.write(annot_save+fileName + ".xml", encoding='utf8')


im = os.listdir(train_)
a = ''
for i in im:
    a+= os.path.basename(i).split('.')[0] + '\n'
with open(split_save+'trainval.txt', "w") as f:
    f.write(a)
    
im = os.listdir(test_)
a = ''
for i in im:
    a+= os.path.basename(i).split('.')[0] + '\n'
with open(split_save+'test.txt', "w") as f:
    f.write(a)

print('XML annotations have been created')
#shutil.copy(train_, image_save)
#shutil.copy(test_, image_save)

#for i in glob(f"{image_save}*.JPG"):
 #   os.system(f"cp {i} {i.split('.')[0]+'.jpg'}")
    
os.mkdir('data/dataset/train')
os.mkdir('data/dataset/test')

train_images = os.listdir(train_)
test_images = os.listdir(test_)

train = 'data/dataset/train/'
for img in train_images:
    file_name = img.split('.')[0]
    xml_file = file_name+'.xml'
    #print(xml_file)
    shutil.move(image_save+img,train)
    shutil.move(annot_save+xml_file,train)
print('Train data(images and annotations) have been moved to train folder')
    
test = 'data/dataset/test/'
for img in test_images:
    file_name = img.split('.')[0]
    xml_file = file_name+'.xml'
    #print(xml_file)
    shutil.move(image_save+img,test)
    shutil.move(annot_save+xml_file,test)
print('Test data(images and annotations) have been moved to Test folder')


all_data = pd.read_csv(path)
print('Data read from the downloaded Annotation.csv file')

all_data.columns =['image','x_min','y_min','x_max','y_max','class']

all_data['class'] = 'b'

test_data = []
test_images = os.listdir(test_)
for img in test_images:
    for i in range(len(all_data)):
        if all_data['image'][i] == img:
            test_data.append([all_data.loc[i][0],all_data.loc[i][1],all_data.loc[i][2],all_data.loc[i][3],all_data.loc[i][4],all_data.loc[i][5]])
            
data = pd.DataFrame(test_data,columns = ['image','x_min','y_min','x_max','y_max','class'])

data.to_csv('data/test_data.csv')
print('Test CSV file has been created')