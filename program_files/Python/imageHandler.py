import PIL
from PIL import Image
from cv2 import cv2
import numpy as np
import random
import os
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
import skimage.io
from skimage import img_as_ubyte
from pathlib import Path

class ImageHandler:

    def __init__(self, _imagePath = ""):
        self.imagePath = _imagePath

        self.imageArray = None


    def crop(self, path, input, height, width, k, page, area):
        im = Image.open(input)
        imgwidth, imgheight = im.size
        for i in range(0,imgheight,height):
            for j in range(0,imgwidth,width):
                box = (j, i, j+width, i+height)
                a = im.crop(box)
                try:
                    o = a.crop(area)
                    o.save(os.path.join(path,"PNG","%s" % page,"IMG-%s.png" % k))
                except:
                    pass


    def random_rotation(self, image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
        random_degree = random.uniform(-25, 25)
        return sk.transform.rotate(image_array, random_degree)

    def random_noise(self, image_array: ndarray):
        # add random noise to the image
        return sk.util.random_noise(image_array)

    def horizontal_flip(self, image_array: ndarray):
        # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
        return image_array[:, ::-1]

     
            
    '''
    Method used to generate images in a 28 * 28 size, 
    with gray scales.

    '''        
    def generateData(self):
        # images path
        #images_path = 'input/'
        images_path = 'RawData/'
        output_path = 'Data/'

        # resize parameters
        width_size = 28
        height_size = 28

        # resize process
        images = os.listdir(images_path)
        resized_images = []
        flattened_images = []

        # operations
        for i in images:
            current_image = images_path + i
            # open the image in grayscale
            image_array = cv2.imread(current_image, 0)
            # resize image
            resized_image = cv2.resize(image_array, (width_size, height_size), cv2.INTER_LINEAR)
            # add results
            resized_images.append(resized_image)
            flattened_images.append(resized_image.flatten())

            cv2.imwrite(output_path + i , resized_image)

            #safe image in txt
            #with open("data.txt", 'a') as f:
            #    self.saveArrayIntoFile(resized_image.flatten(), f)
            


        '''
        Given RRDA_dataA

        Returns a 5 element array, => [ 1 0 0 0 0] which means, A.

        '''
    def getType(self, string):
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





    def saveArrayIntoFile(self, array, f):
        for item in array:
            f.write("%s " % item)
        f.write("\n")
        
    
    
    ''' 
    Method used to AUG data for Neural Network

    '''
    def generateMoreData(self):
        # our folder path containing some images
        
        folder_path = Path('Data')

        
        # new folder path
        folder_path_aug = Path('AugData')

        # the number of file to generate
        num_files_desired = 10

        # loop on all files of the folder and build a list of files paths
        images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        imageNames = os.listdir(folder_path)

        num_generated_files = 0
        
        counterName = 0

        for image in images:
            num_generated_files = 0

            while num_generated_files <= num_files_desired:
                print("Generating: ", num_generated_files)
                # random image from the folder
                image_path = image

                # read image as an two dimensional array of pixels
                image_to_transform = sk.io.imread(image_path)

                transformed_image = self.random_rotation(image_to_transform)
                
                # define a name for our new file
                #new_file_path = '%s\\TEST_%s%s.jpg' % (folder_path_aug, imageNames[counterName].rstrip('.jpg') ,num_generated_files)
                #new_file_path = '%s\\RRDA_%s%s.jpg' % (folder_path_aug, imageNames[counterName] ,num_generated_files)
                new_file_path = '%s%s.jpg' % (image_path.rstrip('.jpg'), num_generated_files)


                # write image to the disk
                sk.io.imsave(new_file_path, img_as_ubyte(transformed_image))
                num_generated_files += 1
            counterName += 1



    '''
        Method used to generate data without noise or rotations

    '''
    def generateMoreDataSimple(self):
        # our folder path containing some images
        
        folder_path = Path('Data')

        # the number of file to generate
        num_files_desired = 50

        # loop on all files of the folder and build a list of files paths
        images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        num_generated_files = 0
        
        counterName = 0

        for image in images:
            num_generated_files = 0

            while num_generated_files <= num_files_desired:
                print("Generating: ", num_generated_files)
                # random image from the folder
                image_path = image

                # read image as an two dimensional array of pixels
                image_to_transform = sk.io.imread(image_path)

                #transformed_image = self.random_rotation(image_to_transform)
                
                # define a name for our new file
                new_file_path = '%s%s.jpg' % (image_path.rstrip('.jpg'), num_generated_files)


                # write image to the disk
                sk.io.imsave(new_file_path, img_as_ubyte(image_to_transform))
                num_generated_files += 1
            counterName += 1


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








img_hand = ImageHandler()
# first generate data for aug
img_hand.generateData()

# second, generate more sample images
img_hand.generateMoreDataSimple()




