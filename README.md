Dataset preparation steps:

    Here I have done following steps to prepare my dataset.
    step1. First downloading all the required files and images from the link.

    step2. Pre-process the data in the required VOC format.
        --Annotated xml file is created in: data/dataset/Annotatinos.
        --Train and test split is created as given for the task: data/dataset/ImageSets/Main.
        --All the images are put in one folder: data/dataset/JPEGImages.

    step3. Then I have moved train data(annotations and images) to seperate folder: data/dataset/train/
            and 
            Then I have moved test data(annotations and images) to seperate folder:data/dataset/test/

    step4. Then I have readed the annotations from file: data/grocerydataset/annotations.csv for the test images and copy those test annotatios to file: data/test_data.csv for validation purpose.

    NOTE: Apart from the default data augmentation (normalizing the image, with mean and std), I did not use any augmented data



Detection network used: 

    Here I have used SSD Model : "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8" from Tensorflow Object Detection APIs.

    -- Total number of classes are 2: one for the product and one for the backgorund class (no product).
    -- For the anchor box: As mentioned in the problem statemnt only 1 anchor box per feture map is allowed. So we use only one anchor map here: [0.5](comman portrait anchor box for all the objects), as I have found all the objects present in our data is portrait shape.
    -- training-parameters/hyper-parameters and anchor box tuning: Here I have used default parameters except epochs : 12000(after many observations) and batch-size :4(as per system strength) 
    and for anchor-box:[0.5] as all the images are in portrait shape.




Q&A: Detailed answers to the following questions:

    What is the purpose of using multiple anchors per feature map cell?
    -- The purpose of of using multiple anchors per feature map cell is that, so our model can detect and consider objects with different shapes in an image. Suppose an image consist of a ball and a person and in that image our model would need different anchor boxes shapes to correctly predict the bbox.

    Does this problem require multiple anchors? Please justify your answer.
    -- Well no it dosn't require multiple anchors because as I have noticed the shape of all objects in the images are of similar shape(portrait shape). So we have used only one anchor box: [0.5] here.


