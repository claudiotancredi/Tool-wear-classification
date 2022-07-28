import glob
import shutil
import os
import json
import numpy as np
import sys
import matplotlib.pyplot as plt
import sklearn.metrics as met

# Root directory of the project
ROOT_DIR = os.path.abspath("/content/mlinapptests/src/models/Baseline_Mask_RCNN/maskrcnn")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn import utils

def prepare_images_and_folder(unpack=True):
    """Prepare dataset folder, unpack if requested, prepare annotations
    and images folders."""
    print("Working in Baseline directory...")
    ds_path = "/content/drive/MyDrive/2022-DC03/dataset"
    if not os.path.exists(ds_path):
        os.makedirs(ds_path)
        print("dataset folder didn't exist and has been created")
    if unpack and len(glob.glob("/content/drive/MyDrive/2022-DC03/dataset/*.bmp"))==0:
        print("Unpacking images to dataset folder...")
        shutil.unpack_archive("/content/drive/MyDrive/2022-DC03/images.zip", "/content/drive/MyDrive/2022-DC03/dataset", "zip")
        print("Number of unpacked images: ", len(glob.glob("/content/drive/MyDrive/2022-DC03/dataset/*.bmp")))
    newpath1 = '/content/drive/MyDrive/2022-DC03/Baseline/annotations' 
    newpath2 = '/content/drive/MyDrive/2022-DC03/Baseline/images'
    newpath3 = '/content/drive/MyDrive/2022-DC03/Baseline/images/train'
    newpath4 = '/content/drive/MyDrive/2022-DC03/Baseline/images/val'
    newpath5 = '/content/drive/MyDrive/2022-DC03/Baseline/images/test'
    if not os.path.exists(newpath1):
        os.makedirs(newpath1)
        print("annotations folder didn't exist and has been created")
    if not os.path.exists(newpath2):
        os.makedirs(newpath2)
        print("images folder didn't exist and has been created")
    if not os.path.exists(newpath3):
        os.makedirs(newpath3)
        print("images/train folder didn't exist and has been created")
    if not os.path.exists(newpath4):
        os.makedirs(newpath4)
        print("images/val folder didn't exist and has been created")
    if not os.path.exists(newpath5):
        os.makedirs(newpath5)
        print("images/test folder didn't exist and has been created")
    
def copy_images(dataset):
    """Copy images in train, val and test folders according to the split."""
    ff = open('/content/drive/MyDrive/2022-DC03/Baseline/annotations/'+dataset+'.json', "r")
    data = json.load(ff)
    images = data["images"]
    filenames = []
    for image in images:
      filenames.append(image["file_name"])
    print("Number of images for ", dataset,": ", len(filenames))
    for f in filenames:
      shutil.copyfile("/content/drive/MyDrive/2022-DC03/dataset/" + f, "/content/drive/MyDrive/2022-DC03/Baseline/images/"+dataset+"/" + f)
    ff.close()
    
def visualize_random_samples(dataset, n):
    image_ids = np.random.choice(dataset.image_ids, n)
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset.class_names)
        
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def visualize_gt(dataset, inference_config, image_id=None):
    if image_id==None:
        image_id=np.random.choice(dataset.image_ids)
    print("Image name: ", dataset.load_name(image_id))
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, inference_config, 
                           image_id)
    
    modellib.log("original_image", original_image)
    modellib.log("image_meta", image_meta)
    modellib.log("gt_class_id", gt_class_id)
    modellib.log("gt_bbox", gt_bbox)
    modellib.log("gt_mask", gt_mask)
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset.class_names, figsize=(8, 8))
    return image_id
    
def visualize_pred(dataset, inference_config, image_id, model):
    print("Image name: ", dataset.load_name(image_id))
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, inference_config, 
                               image_id)
    results = model.detect([original_image], verbose=1)
    
    r = results[0]
    
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'], ax=get_ax())
    
def compute_map(dataset, inference_config, model):
    APs = []
    for i in range(len(dataset.image_ids)):
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, inference_config,
                                   dataset.image_ids[i])
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
        
    print("mAP: %.3f" % np.mean(APs))
    
def compute_accuracy(dataset, inference_config, model):
    n_images = len(dataset.image_ids)
    correct_predictions = 0
    n_ok = 0
    n_nok = 0
    n_doubt = 0
    n_critical = 0
    n_semicritical=0
    correct_ok=0
    correct_nok=0
    correct_doubt=0
    for i in range(len(dataset.image_ids)):    # iterate over all the ids (all the images) for the test set
      masks_areas=[]  # list of values for the areas of the different masks of the i-th image
      original_image = modellib.load_image_gt(dataset, inference_config, 
                               dataset.image_ids[i])[0]
      r = model.detect([original_image])[0] # [0] already get the object that we need to work with
      masks = r["masks"]  # extract predicted masks
      for j in range(masks.shape[2]): # masks are in the shape of (320 x 320 x number_of_masks), so iterate over the third dimension and sum all the values of the other two.
        # The sum will give the number of True occurrencies of the mask, and it will be used to identify the biggest mask, as if it is a measure of an area
        masks_areas.append(masks[:,:,j].sum())
      label=""
      max_index = 0
      if (len(masks_areas)>0):
        max_area = max(masks_areas) # find the max value between masks areas
        max_index = masks_areas.index(max_area) # get the index of the max value
        if r["class_ids"][max_index]==1:
          label="wd"
        elif r["class_ids"][max_index]==2:
          label="wn"
        elif r["class_ids"][max_index]==3:
          label="wo"
      else:
        label="wo"
    
      label_gt = dataset.load_name(dataset.image_ids[i]).split('_')[4]
      if label_gt == "wo":
        n_ok+=1
      elif label_gt =="wd":
        n_doubt+=1
      else:
        n_nok+=1
      if label==label_gt:
        correct_predictions+=1
        if label == "wo":
            correct_ok+=1
        elif label == "wd":
            correct_doubt+=1
        else:
            correct_nok+=1
      else:
        if label_gt=="wn" and label=="wo":
            n_critical+=1
        elif label_gt=="wn" and label=="wd":
            n_semicritical+=1
        print("label_gt is: ", label_gt, " but predicted label is: ", label)
    print("\nNumber of images in test set: ", n_images)
    print("Number of OK images in test set: ", n_ok)
    print("Number of NOK images in test set: ", n_nok)
    print("Number of DOUBT images in test set: ", n_doubt)
    print("\nNumber of critical wrong predictions (gt is wn, prediction is wo): ", n_critical)
    print("\nNumber of semi-critical wrong predictions (gt is wn, prediction is wd): ", n_semicritical)
    print("\nNumber of correct predictions: ", correct_predictions)
    print("Accuracy: ", float(correct_predictions)/n_images)
    print("\nAccuracy for class OK: ", float(correct_ok)/n_ok)
    print("Accuracy for class NOK: ", float(correct_nok)/n_nok)
    print("Accuracy for class DOUBT: ", float(correct_doubt)/n_doubt)
   
