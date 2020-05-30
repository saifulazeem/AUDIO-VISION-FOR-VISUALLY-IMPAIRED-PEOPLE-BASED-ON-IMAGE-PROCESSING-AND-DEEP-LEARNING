import matplotlib
import tensorflow as tf
print(tf.version)
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
##from .utils import ops as utils_ops
#from ..object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')
# This is needed to display the images.
#%matplotlib #inline
from ..object_detection.utils import label_map_util

from ..object_detection.utils import visualization_utils as vis_util
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('C:/flask_work/projects/objectdet/TensorFlow/models/research/object_detection/data', 'mscoco_label_map.pbtxt')
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd()) #current working directory
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def obj_det():

  import pathlib

  # For the sake of simplicity we will use only 2 images:
  # image1.jpg
  # image2.jpg
  # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
  PATH_TO_TEST_IMAGES_DIR = pathlib.Path(
    'C:/flask_work/projects/')

  TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))

  IMAGE_SIZE = (12, 8)
  #now/TEST_IMAGE_PATHS

  #import cv2, time

  #with detection_graph.as_default():
  #  with tf.compat.v1.Session(graph=detection_graph) as sess:
  #    sess.run(tf.compat.v1.global_variables_initializer())
  #    img = 1
  #    for image_path in TEST_IMAGE_PATHS:
  #      image = Image.open(image_path)
  #      image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  #      image_np_expanded = np.expand_dims(image_np, axis=0)
  #      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
  #      # Each box represents a part of the image where a particular object was detected.
  #      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
  #      scores = detection_graph.get_tensor_by_name('detection_scores:0')
  #      classes = detection_graph.get_tensor_by_name('detection_classes:0')
  #      num_detections = detection_graph.get_tensor_by_name('num_detections:0')

  #      (boxes, scores, classes, num_detections) = sess.run(
  #        [boxes, scores, classes, num_detections],
  #        feed_dict={image_tensor: image_np_expanded})

  #      out = vis_util.visualize_boxes_and_labels_on_image_array(
  #        image_np,
  #        np.squeeze(boxes),
  #        np.squeeze(classes).astype(np.int32),
  #        np.squeeze(scores),
  #        category_index,
  #        use_normalized_coordinates=True,
  #        line_thickness=8,
  #        min_score_thresh=0.5)

  #      # ymin = int((boxes[0][0][0]*image_height))
  #      # xmin = int((boxes[0][0][1]*image_width))
  #      # ymax = int((boxes[0][0][2]*image_height))
  #      # xmax = int((boxes[0][0][3]*image_width))
  #      # (left, right, top, bottom) = (xmin * image_width, xmax * image_width, ymin * image_height, ymax * image_height)
  #      # Result = np.array(image_np[ymin:ymax,xmin:xmax])
  #      # img_item = "C:/Users/Bunny/Documents/TensorFlow/models/research/object_detection/test_images"
  #      # cv2.imwrite(img_item, Result)

  #      # get objects name from image having score value greater then 50%)
  #      your_list = [category_index.get(value) for index, value in enumerate(classes[0]) if scores[0, index] > 0.3]
  #      name = [item['name'] for item in your_list]  # names of objects in image
  #      print(name)  # rint out names i.e used in making decription
  #      print(len(name))  # lenght of objects , used while making description



  #      im_width, im_height = IMAGE_SIZE
  #      coordinates_list = []
  #      result = []
  #      counter_for = 0
  #      for i, b in enumerate(boxes[0]):
  #        if scores[0, i] > 0.5:
  #          ymin = int((boxes[0][i][0] * im_height))  # top
  #          xmin = int((boxes[0][i][1] * im_width))  # left
  #          ymax = int((boxes[0][i][2] * im_height))  # bottom
  #          xmax = int((boxes[0][i][3] * im_width))  # right
  #          (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
  #                                        ymin * im_height, ymax * im_height)
  #          coordinates_list.append([ymin, ymax, xmin, xmax])
  #          counter_for = counter_for + 1
  #          result.append([(xmin + xmax) / 2])
  #          # middle_col = (xmin + xmax) / 2
  #          # print(middle_row,middle_col)

  #      print(coordinates_list)
  #      print(result)
  #      plt.figure(figsize=IMAGE_SIZE)
  #      plt.imshow(image_np)

  #print(result)
  #print(name)
  #zipped = zip(result, name)
  #list(zipped)

  #for b, n in zip(result, name):
  #  if b >= [0.0] and b <= [4.0]:
  #    print("  the " + n + " is on left side of image")

  #  elif b > [4.0] and b <= [8.0]:
  #    print("  the " + n + " is on center side of image")

  #  elif b > [8.0] and b <= [12.0]:
  #    print("  the " + n + " is on right side of image")

  #  else:
  #    print(' ')
  #print(name)


  #def getDuplicatesWithCount(name):
  #  ''' Get frequency count of duplicate elements in the given list '''
  #  dictOfElems = dict()
  #  # Iterate over each element in list
  #  for elem in name:
  #    # If element exists in dict then increment its value else add it in dict
  #    if elem in dictOfElems:
  #      dictOfElems[elem] += 1
  #    else:
  #      dictOfElems[elem] = 1

  #      # Filter key-value pairs in dictionary. Keep pairs whose value is greater than 1 i.e. only duplicate elements from list.
  #  dictOfElems = {key: value for key, value in dictOfElems.items() if value > 1}
  #  # Returns a dict of duplicate elements and thier frequency count
  #  return dictOfElems


  ## Get a dictionary containing duplicate elements in list and their frequency count
  #dictOfElems = getDuplicatesWithCount(name)

  #for key, value in dictOfElems.items():
  #  print(key, ' :: ', value)


  ###new code starts here

  import cv2, time

  with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        sess.run(tf.global_variables_initializer())
        img = 1
        for image_path in TEST_IMAGE_PATHS:
          image = Image.open(image_path)
          image_np = load_image_into_numpy_array(image)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')

          (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

          out = vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.3)

          # ymin = int((boxes[0][0][0]*image_height))
          # xmin = int((boxes[0][0][1]*image_width))
          # ymax = int((boxes[0][0][2]*image_height))
          # xmax = int((boxes[0][0][3]*image_width))
          # (left, right, top, bottom) = (xmin * image_width, xmax * image_width, ymin * image_height, ymax * image_height)
          # Result = np.array(image_np[ymin:ymax,xmin:xmax])
          # img_item = "C:/Users/Bunny/Documents/TensorFlow/models/research/object_detection/test_images"
          # cv2.imwrite(img_item, Result)

          # get objects name from image having score value greater then 50%)
          your_list = [category_index.get(value) for index, value in enumerate(classes[0]) if scores[0, index] > 0.3]
          name = [item['name'] for item in your_list]  # names of objects in image
          #now/print(name)  # print out names i.e used in making decription
          var_len = len(
            name)  # lenght of objects , used while making description #this variable is used in description to tell number of objects

          im_width, im_height = IMAGE_SIZE
          coordinates_list = []
          result = []
          counter_for = 0
          for i, b in enumerate(boxes[0]):
            if scores[0, i] > 0.3:
              ymin = int((boxes[0][i][0] * im_height))  # top
              xmin = int((boxes[0][i][1] * im_width))  # left
              ymax = int((boxes[0][i][2] * im_height))  # bottom
              xmax = int((boxes[0][i][3] * im_width))  # right
              (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                            ymin * im_height, ymax * im_height)
              coordinates_list.append([ymin, ymax, xmin, xmax])
              counter_for = counter_for + 1
              result.append([(xmin + xmax) / 2])
              # middle_col = (xmin + xmax) / 2
              # print(middle_row,middle_col)

          #now/print(coordinates_list)
          #now/print(result)
          #now/plt.figure(figsize=IMAGE_SIZE)
          #now/plt.imshow(image_np)
          #now/print(var_len)

        # copy paste this cell again
  #RelationShip Module Integration
  # # new chunk implement relation with comparing every coordinate box with each other
  # # next code cell is alos its part (a)
  # class Solution(object):
  #   def isRectangleOverlap(self, rec1, rec2):
  #     if (rec1[2] <= rec2[0]):
  #       print(n + " beside " + m)
  #       praposition=n + " beside " + m
  #       return praposition
  #
  #     elif (rec1[3] <= rec2[1]):
  #       print(n + " is below" + m)
  #       praposiation=n + " is below" + m
  #       return praposiation
  #
  #     elif (rec1[0] >= rec2[2]):
  #       print(n + " beside " + m)
  #       praposiation=n + " beside " + m
  #       return praposiation
  #
  #
  #     elif (rec1[1] >= rec2[3]):
  #       print(n + " is above  " + m)
  #       praposiation = n + " is above" + m
  #       return praposiation
  #
  #
  #     else:
  #       print(n + " in front of " + m)
  #       praposiation = n + " in front of " + m
  #       return praposiation
  #
  # ##new chunk related to relation comparing each boz part (b)
  # print(coordinates_list)
  # print(name)
  #
  # for i, n in zip(coordinates_list, name):
  #   rec1 = i
  #   # print(rec1)
  #   for j, m in zip(coordinates_list, name):
  #     if (i == j):
  #       print("Cant Identify Relationship")
  #       #rec2 =! j
  #
  #     else:
  #       rec2 = j
  #
  #       p = n + " and " + m
  #       p2 = n
  #       p3 = m
  #       # print(p)
  #       x = Solution()
  #       x.isRectangleOverlap(rec1, rec2)

  # new chunk to find mid point. and find out relation ith respect  to that central coordinate
  # part(a)
  midd_point_list = []

  def findMiddle(coordinates_list):
    if len(coordinates_list)>0:
      middle = float(len(coordinates_list)) / 2
      if middle % 2 != 0:
        mid = coordinates_list[int(middle - .5)]
        midd_point_list.append(mid)
        return mid
      else:
        mid = (coordinates_list[int(middle)])
        midd_point_list.append(mid)
        return mid
    else:
      return "No RelationShip Found"

  findMiddle(coordinates_list)

  # new chunk
  # part(b)
  class Solution(object):
    def isRectangleOverlap(self, rec1, rec2):
      if (rec1[2] <= rec2[0]):
        print(n + " beside " + m)
        prep = n + " beside " + m
        return prep

      elif (rec1[3] <= rec2[1]):
        print(n + " is below" + m)
        prep = n + " is below" + m
        return prep

      elif (rec1[0] >= rec2[2]):
        print(n + " beside " + m)
        prep = n + " beside " + m
        return prep

      elif (rec1[1] >= rec2[3]):
        print(n + " is above  " + m)
        prep = n + " is above  " + m
        return prep

      else:
        print(n + " in front of " + m)
        prep = n + " in front of " + m
        return prep

  # part(c)
  relationshhip_list = []
  for i, n in zip(midd_point_list, name):
    rec1 = i
    # print(rec1)
    for j, m in zip(coordinates_list, name):
      if (i == j):
        print("Cant Identify Relationship")
        #rec2 != j
      else:
        rec2 = j

        p = n + " and " + m
        p2 = n
        p3 = m
        # print(p)
        x = Solution()
        rslt= x.isRectangleOverlap(rec1, rec2)
        relationshhip_list.append(rslt)



        # from here go to last code cell "decription "
  #RelationShip Integration Ends Here..

  #now/print(result)
  #now/print(name)
  print("Rel_list",relationshhip_list)
  if len(relationshhip_list)==0:
    rslt="No RelationShip is Identified"
    relationshhip_list.append(rslt)
  zipped = zip(result, name)
  list(zipped)

  dictOfElems = dict()
  counter1 = 0
  l = []
  for b, n in zip(result, name):
    if b >= [0.0] and b <= [4.0]:
      left_obj = "  the " + n + " is on left side of image"
      #now/print(left_obj)

    elif b > [4.0] and b <= [8.0]:
      cen_obj = "  the " + n + " is on center side of image"
      #now/print(cen_obj)

    elif b > [8.0] and b <= [12.0]:
      right_obj = "  the " + n + " is on right side of image"
      #now/print(right_obj)

    else:
      print(' ')



  #here putting objects that are in right, left and center, into seperate list i.e left_item_list)  and from this further will count the deplicate object in each  list i.e 2 person, 3 car
  left_item_list = []
  center_item_list =[]
  right_item_list =[]
  for b,n in zip(result,name):
      if b >= [0.0] and  b <= [4.0]:
          left_item_list.append(n)
      elif b > [4.0]  and b <= [8.0]:
          center_item_list.append(n)
      elif b > [8.0]  and b <= [12.0]:
           right_item_list.append(n)
  #/now/print(left_item_list)
  #/now/print(center_item_list)
  #now/print(right_item_list)

  #now/print(name)


  def getDuplicatesWithCount(name):
    ''' Get frequency count of duplicate elements in the given list '''
    dictOfElems = dict()
    # Iterate over each element in list
    for elem in name:
      # If element exists in dict then increment its value else add it in dict
      if elem in dictOfElems:
        dictOfElems[elem] += 1
      else:
        dictOfElems[elem] = 1

        # Filter key-value pairs in dictionary. Keep pairs whose value is greater than 1 i.e. only duplicate elements from list.
    dictOfElems = {key: value for key, value in dictOfElems.items() if value > 1}
    # Returns a dict of duplicate elements and thier frequency count
    return dictOfElems


  # Get a dictionary containing duplicate elements in list and their frequency count
  dictOfElems = getDuplicatesWithCount(name)
  total_obj_count = []
  for key, value in dictOfElems.items():
    co_item = key, ' :: ', value
    total_obj_count.append(str(value) + ' are ' + str(key) + ', ')

  #now/print(total_obj_count)


  # this is for counting deplicate object ob left side of image
  def getDuplicatesWithCount(left_item_list):
    ''' Get frequency count of duplicate elements in the given list '''
    dictOfElems = dict()
    # Iterate over each element in list
    for elem in left_item_list:
      # If element exists in dict then increment its value else add it in dict
      if elem in dictOfElems:
        dictOfElems[elem] += 1
      else:
        dictOfElems[elem] = 1

        # Filter key-value pairs in dictionary. Keep pairs whose value is greater than 1 i.e. only duplicate elements from list.
    dictOfElems = {key: value for key, value in dictOfElems.items() if value >= 1}
    # Returns a dict of duplicate elements and thier frequency count
    return dictOfElems


  # Get a dictionary containing duplicate elements in list and their frequency count
  dictOfElems = getDuplicatesWithCount(left_item_list)
  left_count_obj = []
  for key, value in dictOfElems.items():
    co_item = key, ' :: ', value
    #left_count_obj.append(str(value) + ' ' + str(key) + '')
    if value >= 2:  # $
      left_count_obj.append('few ' + str(key) + '')  # $
    if value < 2:  # $
      left_count_obj.append(str(value) + ' ' + str(key) + '')  # $

    #now/print(left_count_obj)


  # this is for counting deplicate object ob center side of image
  def getDuplicatesWithCount(center_item_list):
    ''' Get frequency count of duplicate elements in the given list '''
    dictOfElems = dict()
    # Iterate over each element in list
    for elem in center_item_list:
      # If element exists in dict then increment its value else add it in dict
      if elem in dictOfElems:
        dictOfElems[elem] += 1
      else:
        dictOfElems[elem] = 1

        # Filter key-value pairs in dictionary. Keep pairs whose value is greater than 1 i.e. only duplicate elements from list.
    dictOfElems = {key: value for key, value in dictOfElems.items() if value >= 1}
    # Returns a dict of duplicate elements and thier frequency count
    return dictOfElems


  # Get a dictionary containing duplicate elements in list and their frequency count
  dictOfElems = getDuplicatesWithCount(center_item_list)
  center_count_obj = []
  for key, value in dictOfElems.items():
    co_item = key, ' :: ', value
    #center_count_obj.append(str(value) + ' ' + str(key) + '')
    if value >= 2:  # $
      center_count_obj.append('few ' + str(key) + '')  # $

      # now/print(center_count_obj)
    if value < 2:  # $
      center_count_obj.append(str(value) + ' ' + str(key) + '')  # $

    #now/print(center_count_obj)


  # this is for counting deplicate object ob right side of image
  def getDuplicatesWithCount(right_item_list):
    ''' Get frequency count of duplicate elements in the given list '''
    dictOfElems = dict()
    # Iterate over each element in list
    for elem in right_item_list:
      # If element exists in dict then increment its value else add it in dict
      if elem in dictOfElems:
        dictOfElems[elem] += 1
      else:
        dictOfElems[elem] = 1

        # Filter key-value pairs in dictionary. Keep pairs whose value is greater than 1 i.e. only duplicate elements from list.
    dictOfElems = {key: value for key, value in dictOfElems.items() if value >= 1}
    # Returns a dict of duplicate elements and thier frequency count
    return dictOfElems


  # Get a dictionary containing duplicate elements in list and their frequency count
  dictOfElems = getDuplicatesWithCount(right_item_list)
  right_count_obj = []
  for key, value in dictOfElems.items():
    co_item = key, ' :: ', value
    #right_count_obj.append(str(value) + ' ' + str(key) + '')
    if value >= 2:  # $
      right_count_obj.append('few ' + str(key) + '')  # $

    if value < 2:  # $
      right_count_obj.append(str(value) + ' ' + str(key) + '')

    #now/print(right_count_obj)

  if var_len == 0:
    a = 'there is no '
    description = " I can not really describe the scene but it seems like "  # $ if nothing is detected.


  elif var_len > 0:
    a = 'there are almost ' + str(var_len) + ' '
    if len(right_count_obj) != 0 and len(left_count_obj) != 0 and len(center_count_obj) != 0:  # $

      description = " I can see " + ''.join(left_count_obj) + " at left side, " + ''.join(
        center_count_obj) + " in the center and " + ''.join(right_count_obj) \
                    + " at the right side."+" The Relationship between objects are "+str(relationshhip_list)  # $ if image contain objects

    elif len(right_count_obj) == 0 and len(left_count_obj) != 0 and len(center_count_obj) != 0:  # $

      description = " It seems like " + ''.join(left_count_obj) + " at left side and" + ''.join(
        center_count_obj) + " in the center."+" The Relationship between objects are "+str(relationshhip_list)  # $ if image contain objects #$

    elif len(right_count_obj) != 0 and len(left_count_obj) == 0 and len(center_count_obj) != 0:  # $

      description = " It seems like " + ''.join(center_count_obj) + " in the center and" + ''.join(right_count_obj) \
                    + " at the right side."+" The Relationship between objects are "+str(relationshhip_list)  # $ if image contain objects

    elif len(right_count_obj) != 0 and len(left_count_obj) != 0 and len(center_count_obj) == 0:  # $

      description = " It seems like " + ''.join(left_count_obj) + " at the left side and" + ''.join(right_count_obj) \
                    + " at the right side."+" The Relationship between objects are "+str(relationshhip_list)  # $ if image contain objects

    elif len(right_count_obj) == 0 and len(left_count_obj) == 0 and len(center_count_obj) != 0:  # $

      description = " It look like " + ''.join(center_count_obj) + " in the center."+" The Relationship between objects are "+str(relationshhip_list)  # $ if image contain objects

    elif len(right_count_obj) != 0 and len(left_count_obj) == 0 and len(center_count_obj) == 0:  # $

      description = " It can see " + ''.join(right_count_obj) + " at the right side."+" The Relationship between objects are "+str(relationshhip_list)  # $ if image contain objects

    elif len(right_count_obj) == 0 and len(left_count_obj) != 0 and len(center_count_obj) == 0:  # $

      description = " I can see " + ''.join(center_count_obj) + " at the left side."+" The Relationship between objects are "+str(relationshhip_list)  # $ if image contain objects

  des = " " + str(a) + " objects identified in given image, among them " + ''.join(
    total_obj_count) + " moreover " + ''.join(left_count_obj) + " at left side and, " + ''.join(
    center_count_obj) + ' are at center and  ' + ''.join(right_count_obj) + " are at right side of image Now The Relationshipes between" \
                                                                            " objects are "+str(relationshhip_list)
  print(des)
  print(description)
  return description

obj_des=obj_det()

print(obj_des)







def abc():
  x=" Hello"
  return x

myvar=abc()

print(myvar)


