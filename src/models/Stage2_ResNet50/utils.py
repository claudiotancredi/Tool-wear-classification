import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import glob
import random
import PIL
from matplotlib.patches import Rectangle
import sklearn.metrics as met


def display_images(images, titles=None, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    cols = 5
    rows = len(images) //cols +1
    plt.figure(figsize=(18, 18* rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=13)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()
    print("")


def display(images, labels):
    """Display the given image and the top few class masks."""
    titles = []
    for i in range(len(images)):
      titles.append("H x W={}x{} {}".format(images[i].shape[0], images[i].shape[1], labels[i]))
    display_images(images, titles=titles, cmap="Blues_r")


def visualize_n_random_samples(folder, n):
    images_paths = list(glob.glob(folder + '/*/*.jpg'))
    indexes = random.sample(range(0,len(images_paths)), n)
    images=[]
    labels=[]
    for i in range(n):
      images.append(np.array(PIL.Image.open(images_paths[indexes[i]]).convert('RGB')))
      if "wd" in images_paths[indexes[i]]:
        labels.append("GT: DOUBT")
      elif "wo" in images_paths[indexes[i]]:
        labels.append("GT: OK")
      else:
        labels.append("GT: NOK")
    display(images, labels)


def visualize_after_data_augm(train_ds, data_augmentation):
    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images, training=True)
            augmented_images = tf.keras.applications.resnet.preprocess_input(augmented_images, data_format=None)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")


def plot_history(history):
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def find_last(MODEL_DIR, tool):
    """Finds the last checkpoint file of the last trained model in the
    model directory.
    Returns:
        The path of the last checkpoint file
    """
    # Get directory names. Each directory corresponds to a model
    dir_names = next(os.walk(MODEL_DIR))[1]
    key = "tool_wear_"+tool + "_STAGE2_RESNET50_"
    dir_names = filter(lambda f: f.startswith(key), dir_names)
    dir_names = sorted(dir_names)
    if not dir_names:
        import errno
        raise FileNotFoundError(
            errno.ENOENT,
            "Could not find model directory under {}".format(MODEL_DIR))
    # Pick last directory
    dir_name = os.path.join(MODEL_DIR, dir_names[-1])
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("resnet50"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        import errno
        raise FileNotFoundError(
            errno.ENOENT, "Could not find weight files in {}".format(dir_name))
    checkpoint = os.path.join(dir_name, checkpoints[-1])
    print(checkpoint)
    return checkpoint


def display_images_bbox(images, titles=None, cmap=None, norm=None,
                   interpolation=None, bbox=None, names=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    cols = 5
    rows = len(images) //cols +1
    plt.figure(figsize=(30, 30* rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=15)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        plt.text(0, 340, names[i-1], fontsize=12)
        if bbox:
          plt.gca().add_patch(Rectangle((bbox[i-1][0],bbox[i-1][1]),bbox[i-1][2],bbox[i-1][3],linewidth=1,edgecolor='r',facecolor='none'))
        i += 1
    plt.show()
    print("")


def display_bbox(images, labels, bbox, names):
    """Display the given image and the top few class masks."""
    titles = []
    for i in range(len(images)):
      titles.append("H x W={}x{} {}".format(images[i].shape[0], images[i].shape[1], labels[i]))
    display_images_bbox(images, titles=titles, cmap="Blues_r", bbox=bbox, names=names)


def predict_stats(model, test_ds, data):
    predictions=[]
    gt=[]
    for images, labels in test_ds:
        predictions.append([test_ds.class_names[index] for index in np.argmax(model.predict(images),1)])
        l = len(labels)
        g = []
        for i in range(l):
          g.append(labels.numpy()[i])
        gt.append([test_ds.class_names[index] for index in g])

    counter_crit=0
    counter_semicrit=0
    counter_ok=0
    counter_nok=0
    counter_doubt=0
    for i in range(len(gt)):
      for j in range(len(gt[i])):
        if gt[i][j]!=predictions[i][j]:
          if gt[i][j]=="NOK" and predictions[i][j]=="OK":
            counter_crit+=1
          elif gt[i][j]=="NOK" and predictions[i][j]=="DOUBT":
            counter_semicrit+=1
          print("Ground truth was: ", gt[i][j], " but prediction is: ", predictions[i][j])
        else:
          if (gt[i][j]=="NOK"):
            counter_nok+=1
          elif gt[i][j]=="OK":
            counter_ok+=1
          else:
            counter_doubt+=1
    for element in data:
        if len(element["bbox"])==0: # Classified as ok by stage 1
            if "wo" not in element["name"]:
                if "wn" in element["name"]:
                    counter_crit+=1
                gt = ""
                if "wo" in element["name"]:
                  gt="OK"
                elif "wn" in element["name"]:
                  gt = "NOK"
                else:
                  gt = "DOUBT"
                print("Ground truth was: ", gt, " but prediction is: OK")
            else:
                counter_ok+=1

    return counter_ok, counter_doubt, counter_nok, counter_crit, counter_semicrit


def get_prediction_info(model, test_ds_compute, test_ds_global, paths, ORIGINAL_IMAGES_PATH, data, n=10):
    images_arr=[]
    predictions_arr=[]
    bbox_arr=[]
    i = 0

    for images, labels in test_ds_compute:
      for j in range(len(images)):
        if (j+i*4<n):
          im = PIL.Image.open(ORIGINAL_IMAGES_PATH + paths[j+i*4].split("/")[9].split(".")[0] + ".bmp")
          image = np.array(im)
          image = tf.image.resize(image, (320, 320)).numpy()
          images_arr.append(image)
      for j in range(len(images)):
        if (j+i*4<n):
          predictions_arr.append("GT: " + test_ds_global.class_names[labels.numpy()[j]] + " P: " + [test_ds_global.class_names[index] for index in np.argmax([model.predict(images)[j]],1)][0])
      for j in range(len(images)):
        if (j+i*4<n):
          for element in data:
            if element["name"]==paths[j+i*4].split("/")[9].split(".")[0] + ".bmp":
              bbox_arr.append([element["bbox"][1],element["bbox"][0], element["bbox"][3]-element["bbox"][1], element["bbox"][2]-element["bbox"][0]])
      if (len(images)+i*4>=n):
        break
      i=i+1

    paths = paths[:n]
    names = [path.split("/")[9].split(".")[0] + ".bmp" for path in paths]

    return images_arr, predictions_arr, bbox_arr, names

def get_prediction_s1_info(ORIGINAL_IMAGES_PATH, data):
    pred_by_stage1 = []

    for element in data:
      if len(element["bbox"])==0:
        pred_by_stage1.append(element)
    
    images_arr=[]
    predictions_arr=[]
    names = []
    
    for element in pred_by_stage1:
      im = PIL.Image.open(ORIGINAL_IMAGES_PATH + element["name"])
      image = np.array(im)
      image = tf.image.resize(image, (320, 320)).numpy()
      images_arr.append(image)
      gt=""
      if "wo" in element["name"]:
        gt="OK"
      elif "wn" in element["name"]:
        gt = "NOK"
      else:
        gt = "DOUBT"
      predictions_arr.append("GT: " + gt + " P: OK")
      names.append(element["name"])
    return images_arr, predictions_arr, names

def rebalance_train_ds(train_ds):
    dataset_ok = train_ds.unbatch().filter(lambda image, label: label == 2).batch(4)
    dataset_nok = train_ds.unbatch().filter(lambda image, label: label == 1).batch(4)
    dataset_doubt = train_ds.unbatch().filter(lambda image, label: label == 0).batch(4)
    count_ok = len(list(dataset_ok.unbatch().enumerate(start=0).as_numpy_iterator()))
    count_nok = len(list(dataset_nok.unbatch().enumerate(start=0).as_numpy_iterator()))
    count_doubt = len(list(dataset_doubt.unbatch().enumerate(start=0).as_numpy_iterator()))
    print("Number of NOK images in training set: ", count_nok)
    print("Number of DOUBT images in training set: ", count_doubt)
    print("Number of OK images in training set: ", count_ok)
    num_ok_cut = int((count_nok+count_doubt)/2)
    
    train_ds_balanced = dataset_ok.unbatch().take(num_ok_cut).batch(4)
    print("Number of OK images in training set after cut: ", len(list(train_ds_balanced.unbatch().enumerate(start=0).as_numpy_iterator())))
    
    dataset_nok_doubt = dataset_nok.concatenate(dataset_doubt)
    train_ds_balanced = dataset_nok_doubt.concatenate(train_ds_balanced)
    print("Number of images in rebalanced training set: ", len(list(train_ds_balanced.unbatch().enumerate(start=0).as_numpy_iterator())))
    
    train_ds_balanced.class_names = train_ds.class_names
    return train_ds_balanced
