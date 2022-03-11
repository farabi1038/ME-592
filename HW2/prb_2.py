from skimage import data
from skimage import filters
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import cv2 as cv2
import numpy as np

# Sample Image of scikit-image package
img = cv2.imread("/Users/ibnefarabishihab/Desktop/Course materials /ME 592/HW2/eia/1.jpg")
#gray_coffee = rgb2gray(coffee)
print(img.shape)  # Print image shape
#cv2.imshow("original", img)
cropped_image1 = img[10:240, 10:320]
cropped_image2= img[300:535, 320:625]
cropped_image3 = img[590:830, 10:320]

cropped_image4 = img[10:240, 320:625]
cropped_image5= img[300:535, 10:320]
cropped_image6 = img[590:830, 320:625]
#cv2.imshow("cropped1", cropped_image1)
#cv2.imshow("cropped2", cropped_image2)
#cv2.imshow("cropped3", cropped_image3)
#cv2.imshow("cropped4", cropped_image4)
#cv2.imshow("cropped5", cropped_image5)
#cv2.imshow("cropped6", cropped_image6)



# Taking a matrix of size 5 as the kernel

def erode(image,count):
    kernel = np.ones((5, 5), np.uint8)

    img_erosion = cv2.erode(image, kernel, iterations=1)
    img_dilation = cv2.dilate(image, kernel, iterations=1)
    image=image-img_erosion

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    image=cv2.erode(image,horizontal_kernel,iterations=1)

    name="result"+str(count)+".png"
    cv2.imwrite(name, image)




    cv2.imshow('image', cropped_image1)
    #cv2.waitKey(0)
    #plt.show()

erode(cropped_image1,1)
erode(cropped_image2,2)
erode(cropped_image3,3)
erode(cropped_image4,4)
erode(cropped_image5,5)
erode(cropped_image6,6)
