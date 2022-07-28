import glob
import shutil
import os
import json
import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image
import time

# Root directory of the project
ROOT_DIR = os.path.abspath("/content/mlinapptests/src/models/Stage1_Mask_RCNN/maskrcnn")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn import utils

def prepare_folders():
    """Prepare annotations and images folders."""
    print("Working in Stage1_Mask_RCNN directory...")
    newpath1 = '/content/drive/MyDrive/2022-DC03/Stage1_Mask_RCNN/annotations' 
    newpath2 = '/content/drive/MyDrive/2022-DC03/Stage1_Mask_RCNN/images'
    newpath3 = '/content/drive/MyDrive/2022-DC03/Stage1_Mask_RCNN/images/train'
    newpath4 = '/content/drive/MyDrive/2022-DC03/Stage1_Mask_RCNN/images/val'
    newpath5 = '/content/drive/MyDrive/2022-DC03/Stage1_Mask_RCNN/images/test'
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
        
def translate_annotations():
    files = glob.glob("/content/drive/MyDrive/2022-DC03/Baseline/annotations/*.json")
    for i in range(len(files)):
        f = open(files[i], "r")
        data = json.load(f)
        annotations = data["annotations"]
        for annotation in annotations:
            if annotation["category_id"] == 2 or annotation["category_id"] == 3:
                annotation["category_id"] = 1
        data["annotations"] = annotations
        data["categories"] = [{"id":1, "name": "WEAR", "supercategory": "none"}]
        f.close()
        with open('/content/drive/MyDrive/2022-DC03/Stage1_Mask_RCNN/annotations/' + files[i].split("/")[7], "w+") as outf:
            json.dump(data, outf)
        
        
def copy_images(dataset):
    """Copy images in train, val and test folders according to the split."""
    ff = open('/content/drive/MyDrive/2022-DC03/Stage1_Mask_RCNN/annotations/'+dataset+'.json', "r")
    data = json.load(ff)
    images = data["images"]
    filenames = []
    for image in images:
      filenames.append(image["file_name"])
    print("Number of images for ", dataset,": ", len(filenames))
    for f in filenames:
      shutil.copyfile("/content/drive/MyDrive/2022-DC03/dataset/" + f, "/content/drive/MyDrive/2022-DC03/Stage1_Mask_RCNN/images/"+dataset+"/" + f)
    ff.close()
    
def check_num_images():
    print("""Checking if the images in train, val and test folders of the Baseline
          model have been correctly copied for Stage 1 model. It's just a test to see
          if the translation of the annotations worked properly.""")
    error_flag=0
    for set in ["train", "val", "test"]:
        filesBL = glob.glob("/content/drive/MyDrive/2022-DC03/Baseline/images/"+set+"*.bmp")
        filesS1 = glob.glob("/content/drive/MyDrive/2022-DC03/Stage1_Mask_RCNN/images/"+set+"*.bmp")
        for j in range(len(filesBL)):
            filesBL[j]=filesBL[j].split("/")[7]
        for j in range(len(filesS1)):
            filesS1[j]=filesS1[j].split("/")[7]
        for file in filesBL:
            if file not in filesS1:
                error_flag=1
    if error_flag==1:
        print("!!!There is probably something wrong with the translation phase of annotations!!!")
    else:
        print("No errors have been encountered. You can proceed.")
    
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

def extract_bboxes(mask):
    """Compute bounding boxes from masks.

    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])   #THIS IS NOT COCO FORMAT, IT'S THE FORMAT USED BY VISUALIZE.DISPLAY_INSTANCES
    return boxes.astype(np.int32)
    
def visualize_pred(dataset, inference_config, image_id, model, adjusted=False):
    print("Image name: ", dataset.load_name(image_id))
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, inference_config, 
                               image_id)
    results = model.detect([original_image], verbose=1)
    
    r = results[0]
    if adjusted:
        modified_bboxes = extract_bboxes(r['masks'])
        visualize.display_instances(original_image, modified_bboxes, r['masks'], r['class_ids'], 
                            dataset.class_names, r['scores'], ax=get_ax())
    else:
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
    
def process_dataset(dataset, inference_config, model, threshold, visual, ok_path, nok_path, doubt_path):
    detected = []
    not_detected_classified_as_ok = []
    for i in range(len(dataset.image_ids)):    # iterate over all the ids (all the images) for the provided set
      masks_areas=[]  # list of values for the areas of the different masks of the i-th image
      original_image = modellib.load_image_gt(dataset, inference_config, 
                               dataset.image_ids[i])[0]
      r = model.detect([original_image])[0] # [0] already gets the object that we need to work with
      masks = r["masks"]  # extract predicted masks
      for j in range(masks.shape[2]): # masks are in the shape of (320 x 320 x number_of_masks), so iterate over the third dimension and sum all the values of the other two.
        # The sum will give the number of True occurrencies of the mask, and it will be used to identify the biggest mask, as if it is a measure of an area
        masks_areas.append(masks[:,:,j].sum())
      max_index = 0
      found=False
      while (len(masks_areas)>0 and found==False):
        max_area = max(masks_areas) # find the max value between masks areas
        temp_max_index = masks_areas.index(max_area) # get the index of the max value
        if threshold:
            if r['scores'][temp_max_index] >= threshold:
              max_index = temp_max_index
              found=True
            else:
              masks_areas.pop(temp_max_index)
              r['class_ids']=np.delete(r['class_ids'],temp_max_index)
              r['scores']=np.delete(r['scores'],temp_max_index)
        else:
            max_index = temp_max_index
            found=True
      
      if found:
        bbox = [int(v) for v in list(extract_bboxes(masks)[max_index])]
        if masks_areas[max_index]>0: #Area should be >0, otherwise black images will appear. If black
        #images continue to appear, the condition needs to be changed with a different value, that will
        #become a hyperparameter
          detected.append({"name":dataset.load_name(dataset.image_ids[i]),
                       "bbox":bbox}) # save the values in the list.
                       # extract bboxes from masks and then choose the one related to the biggest mask through max_index.
                       # bboxes is a numpy array, which is not serializable into a JSON, so convert to a list and convert its values to int (int32 is not serializable too)
                        #ATTENTION, EXTRACT_BBOXES HAS A DIFFERENT FORMAT FOR BBOXES, USED FOR VISUALIZATION.
                        #IF NEEDED, MANIPULATE THE JSON FILE TO GO BACK TO COCO FORMAT (X1,Y1,WIDTH,HEIGHT)
            
          im = Image.fromarray(original_image)
          im = im.crop((bbox[1], bbox[0], bbox[3], bbox[2]))
          #im = im.resize((120,120), Image.BICUBIC)
          label = dataset.load_name(dataset.image_ids[i]).split('_')[4]
          if i==1 and visual:
              print(label) #Just to check if the split on the string is correct
          if label=="wd":
            im.save(doubt_path+"/"+dataset.load_name(dataset.image_ids[i]).split('.')[0]+".jpg")
          elif label=="wo":
            im.save(ok_path+"/"+dataset.load_name(dataset.image_ids[i]).split('.')[0]+".jpg")
          else:
            im.save(nok_path+"/"+dataset.load_name(dataset.image_ids[i]).split('.')[0]+".jpg")
    
          # IF YOU WANT YOU CAN VISUALIZE THE BIGGEST MASK THAT HAS BEEN CHOSEN.
          if i==1 and visual:
            print("Image name: ", dataset.load_name(dataset.image_ids[i]))
            init = np.zeros((320,320,1)) #the mask has to be 3D even if it's just one
            init[:, :, 0] = r['masks'][:,:,max_index] #copy the biggest mask
            visualize.display_instances(original_image, np.array([extract_bboxes(masks)[max_index]]), init, np.array([r['class_ids'][max_index]]), 
                                      dataset.class_names, np.array([r['scores'][max_index]]), ax=get_ax()) #lots of adjustments to make it work, if you change something
                                      # you should expect a lot of shape errors
        else:
           not_detected_classified_as_ok.append({"name":dataset.load_name(dataset.image_ids[i]),
                        "bbox":[]})                                    
      else:
         not_detected_classified_as_ok.append({"name":dataset.load_name(dataset.image_ids[i]),
                        "bbox":[]})                               
    return detected + not_detected_classified_as_ok, len(detected), len(not_detected_classified_as_ok)
    
def predict_and_prepare_data_for_s2(dataset_train, dataset_val, dataset_test, inference_config, model, tool, threshold=0.95, visual=False):
    newpathslist = ['/content/drive/MyDrive/2022-DC03/stage2/dataset_s1mrcnn_' + tool + "/train",
    '/content/drive/MyDrive/2022-DC03/stage2/dataset_s1mrcnn_'+tool+ "/train"+'/OK',
    '/content/drive/MyDrive/2022-DC03/stage2/dataset_s1mrcnn_'+tool+ "/train"+'/DOUBT',
    '/content/drive/MyDrive/2022-DC03/stage2/dataset_s1mrcnn_'+tool+ "/train"+'/NOK',
    '/content/drive/MyDrive/2022-DC03/stage2/dataset_s1mrcnn_' + tool + "/test",
    '/content/drive/MyDrive/2022-DC03/stage2/dataset_s1mrcnn_'+tool+ "/test"+'/OK',
    '/content/drive/MyDrive/2022-DC03/stage2/dataset_s1mrcnn_'+tool+ "/test"+'/DOUBT',
    '/content/drive/MyDrive/2022-DC03/stage2/dataset_s1mrcnn_'+tool+ "/test"+'/NOK']
    for path in newpathslist:
        if not os.path.exists(path):
            os.makedirs(path)
    detected = [] # list of objects with name of image and bbox as properties. Will be used for the JSON file
    print("Working on training set...")
    _, num, _ = process_dataset(dataset_train, inference_config, model, threshold, visual, newpathslist[1], newpathslist[3], newpathslist[2])
    print("Number of images in training set: ", len(dataset_train.image_ids))
    print("Number of images in training set for which at least one confident (>=", threshold, ") mask has been predicted: ", num)
    print("\nWorking also on validation set...")
    _, num, _ = process_dataset(dataset_val, inference_config, model, threshold, visual, newpathslist[1], newpathslist[3], newpathslist[2])
    print("Number of images in validation set: ", len(dataset_val.image_ids))
    print("Number of images in validation set for which at least one confident (>=", threshold, ") mask has been predicted: ", num)
    print("\nWorking also on test set...")
    detected, num, num_ok = process_dataset(dataset_test, inference_config, model, None, visual, newpathslist[5], newpathslist[7], newpathslist[6])
    print("Number of images in test set: ", len(dataset_test.image_ids))
    print("Number of images in test set for which at least one mask has been predicted: ", num)
    print("Number of images in test set for which no masks have been predicted, and therefore have ",
          "not been cropped and passed down to 2nd stage, but have been only added to the JSON, ",
          "so that the 2nd stage can consider them directly as OK, as if the prediction was entirely ",
          "performed by the 1st stage: ", num_ok)
    if visual:
        print("Example of JSON object: ", detected[1])
    with open("/content/drive/MyDrive/2022-DC03/stage2/output_s1mrcnn_testset_"+tool+".json", "w+") as f: #output of stage 1
      json.dump(detected, f)
