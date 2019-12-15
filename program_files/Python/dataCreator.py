
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
        value = [1, 0, 0, 0, 0]
    elif "dataB" in string:
        value = [0, 1, 0, 0, 0]
    elif "dataC" in string:
        value = [0, 0, 1, 0, 0]
    elif "dataD" in string:
        value = [0, 0, 0, 1, 0]
    elif "dataE" in string:
        value = [0, 0, 0, 0, 1]

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
    images_path = 'ImageCrop/ImageFiles/AugData/'


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
        
        


        batch_file = open(batch_name, mode='a')
        label_file = open(batch_name_labels, mode ='a')

        for i in range(batch_size):
            img_name = batch_img_paths[i]
            img_label = get_label(img_name)

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


def execute_test_image_to_txt():
    # images path
    images_path = 'ImageCrop/ImageFiles/AugDataTest/'


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

        batch_name = "batchTest%i.txt" % (batch_n)
        batch_name_labels = "labelTest%i.txt" % (batch_n)
        
        


        batch_file = open(batch_name, mode='a')
        label_file = open(batch_name_labels, mode ='a')

        for i in range(batch_size):
            img_name = batch_img_paths[i]
            img_label = get_label(img_name)

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

def img_to_data():
    # images path
    images_path = 'ImageCrop/ImageFiles/DataTest/'


    # resize process
    images = os.listdir(images_path)
    batch_n = 0

    # Iterates while all the batches are completed
    for img_name in images:
        
        batch_name = "batch00.txt"
        batch_name_labels = "label00.txt" 
        
        


        batch_file = open(batch_name, mode='a')
        label_file = open(batch_name_labels, mode ='a')

        #img_name = images[i]
        img_label = get_label(img_name)

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

#execute_test_image_to_txt()
