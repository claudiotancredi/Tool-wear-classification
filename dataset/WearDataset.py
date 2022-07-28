import os
import sys
from pycocotools.coco import COCO
import numpy as np
from pycocotools import mask as maskUtils
# Root directory of the project
ROOT_DIR = os.path.abspath("/content/mlinapptests/src/models/Baseline_Mask_RCNN/maskrcnn")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils

class WearDataset(utils.Dataset):
    def load_name(self, image_id):

        image_info = self.image_info[image_id]

        name = image_info['path'].split('/')[-1]
        return name
        
    def load_wear(self, dataset_dir, subset, class_ids=None, return_coco=False, tool="RNGN19", balanced=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        class_ids: If provided, only loads images that have the given classes.
        return_coco: If True, returns the COCO object.
        
        This function is still WIP, it works with a single tool or "all", but
        we should also consider mix of different tools.
        Similarly, balanced is only applied on single tools, for the moment.
        
        """

        coco = COCO(os.path.join(dataset_dir, "annotations", subset + ".json"))
        image_dir = os.path.join(dataset_dir, "images", subset)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("tool_wear", i, coco.loadCats(i)[0]["name"])
        if tool == "all":
          for i in image_ids:
              annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None))
              self.add_image(
                "tool_wear", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=annotations)
        # Add images
        else:
          counter_nok = 0
          counter_doubt = 0
          c_ok=0
          for i in image_ids:
            if tool in (coco.imgs[i]["file_name"]) and "wd" in (coco.imgs[i]["file_name"]):
              counter_doubt+=1
            if tool in (coco.imgs[i]["file_name"]) and "wn" in (coco.imgs[i]["file_name"]):
              counter_nok+=1
            if tool in (coco.imgs[i]["file_name"]) and "wo" in (coco.imgs[i]["file_name"]):
              c_ok+=1
          print("Number of nok: ", counter_nok)
          print("Number of doubt: ", counter_doubt)
          if balanced:
            print("Number of ok before cut: ", c_ok)
          num_ok = (counter_nok+counter_doubt)/2
          counter_ok=0
          for i in image_ids:
            if balanced:
              if tool in (coco.imgs[i]["file_name"]) and "wo" not in (coco.imgs[i]["file_name"]):
                annotations=coco.loadAnns(coco.getAnnIds(
                      imgIds=[i], catIds=class_ids, iscrowd=None))
                self.add_image(
                  "tool_wear", image_id=i,
                  path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                  width=coco.imgs[i]["width"],
                  height=coco.imgs[i]["height"],
                  annotations=annotations)
              elif tool in (coco.imgs[i]["file_name"]) and "wo" in (coco.imgs[i]["file_name"]):
                if counter_ok<num_ok:
                  counter_ok+=1
                  annotations=coco.loadAnns(coco.getAnnIds(
                      imgIds=[i], catIds=class_ids, iscrowd=None))
                  self.add_image(
                    "tool_wear", image_id=i,
                    path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                    width=coco.imgs[i]["width"],
                    height=coco.imgs[i]["height"],
                    annotations=annotations)
            else:
              if tool in (coco.imgs[i]["file_name"]):
                annotations=coco.loadAnns(coco.getAnnIds(
                      imgIds=[i], catIds=class_ids, iscrowd=None))
                self.add_image(
                  "tool_wear", image_id=i,
                  path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                  width=coco.imgs[i]["width"],
                  height=coco.imgs[i]["height"],
                  annotations=annotations)
          if balanced:
            print("Number of ok after cut: ", counter_ok)
          else:
            print("Number of ok without cut because balanced=False: ", c_ok)
        if return_coco:
            return coco

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "tool_wear.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(WearDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "bsd":
            return info["path"]
        else:
            super(WearDataset, self).image_reference(image_id)

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m