
import numpy as np
import os
import pickle
from cv2 import cv2
import random
import array as arr
import skimage as sk
from skimage import transform
from skimage import util
import skimage.io
from skimage import img_as_ubyte
import os
import io

def saveArrayIntoFile(array, f):
    for item in array:
        f.write("%s " % item)
    f.write("\n")
        
def get_label( string):
    value = []
    if "dataA" in string:
        value = [1, 0, 0, 0, 0, 0]
        print("A")
    elif "dataB" in string:
        value = [0, 1, 0, 0, 0, 0]
        print("B")
    elif "dataC" in string:
        value = [0, 0, 1, 0, 0, 0]
        print("C")
    elif "dataD" in string:
        value = [0, 0, 0, 1, 0, 0]
        print("D")

    elif "dataE" in string:
        value = [0, 0, 0, 0, 1, 0]
        print("E")

    elif "DBE-A" in string:
        value = [1, 0, 0, 0, 0, 0]
        print("A")

    elif "DBE-B" in string:
        value = [0, 1, 0, 0, 0, 0]
        print("B")

    elif "DBE-C" in string:
        value = [0, 0, 1, 0, 0, 0]
        print("C")

    elif "DBE-D" in string:
        value = [0, 0, 0, 1, 0, 0]
        print("D")

    elif "E1" in string:
        value = [0, 0, 0, 0, 1, 0]
        print("E")

    elif "F1" in string:
        value = [0, 0, 0, 0, 0, 1]
        print("F")
    return value


def random_elements(array_L, batch_size):
    elements = []
    
    for i in range(batch_size):
        size = len(array_L)
        i  = random.randint(0, size - 1)
        e = array_L[i]
        elements.append(e)
        array_L.remove(e)
    
    return array_L, elements

def img_to_txt_NN():
    # images path
    images_path = 'AugData/'


    # resize process
    images = os.listdir(images_path)
    batch_n = 0
    # Iterates while all the batches are completed
    while len(images) !=  0:
        print(batch_n)
        

        batch_size = 32
        if batch_size > len(images): batch_size = len(images)    
        images, batch_img_paths = random_elements(images, batch_size)

        print("Batch Size", batch_size)

        batch_name = "batch%i.txt" % (batch_n)
        batch_name_labels = "label%i.txt" % (batch_n)
        imgs_name = "file_names%i.txt" % (batch_n)
        


        batch_file = open(batch_name, mode='a')
        label_file = open(batch_name_labels, mode ='a')
        imgs_name_file = open(imgs_name, mode ='a')
        

        for i in range(batch_size):
            img_name = batch_img_paths[i]
            img_label = get_label(img_name)

            # writes name into a file
            imgs_name_file.write(img_name + "\n")

            #image_array = []
            current_image = images_path + img_name
            image_array = cv2.imread(current_image, 0).flatten()
            print(len(image_array))
            
            # writes img into file           batch_array = batch_array + image_array
            #image_array.tofile(batch_file)
            saveArrayIntoFile(image_array, batch_file)


            #adds label to all batch labels
            #batch_lables += img_label
            saveArrayIntoFile(img_label, label_file)

        batch_n += 1
            
        
        batch_file.close()
        label_file.close()


img_to_txt_NN()

