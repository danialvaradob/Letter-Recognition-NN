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



img_hand = ImageHandler()
img_hand.generateMoreData()
#img_hand.generateData()




