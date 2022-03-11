import cv2
import os

import random

path = '/Users/ibnefarabishihab/Desktop/Course materials /ME 592/HW2/eia/leaves/'

os.chdir(os.getcwd())


for i in range(13):
    for j in os.listdir(path):
        name = str(i)+str(j)
        print(name)
        source=path+str(j)
        image = cv2.imread(source)
        height, width = image.shape[:2]
        center = (width/2,height/2)
        rotate_matrix = cv2.getRotationMatrix2D(center = center, angle = random.randint(0,360),scale =random.randint(0,3))
        rotated_image = cv2.warpAffine(src = image, M = rotate_matrix,dsize=(width,height))
        #cv2.imshow(str(i),rotated_image)
        cv2.imwrite(name,rotated_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()



