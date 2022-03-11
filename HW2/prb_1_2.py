import cv2
import os 


print(os.getcwd())
path = '/Users/ibnefarabishihab/Desktop/Course materials /ME 592/HW2/eia/produced/'

os.chdir(path)
print(os.getcwd())


for j in os.listdir(path):
    name = str(j)
    image = cv2.imread(str(j))
    height, width = image.shape[:2]
    print(height,width)
    rotated_image = image[100:200,100:200]
    # center = (width/2,height/2)
    # rotate_matrix = cv2.getRotationMatrix2D(center = center, angle = random.randint(0,360),scale =random.randint(0,3))
    # rotated_image = cv2.warpAffine(src = image, M = rotate_matrix,dsize=(width,height))
    # cv2.imshow(str(j),rotated_image)
    cv2.imwrite(name,rotated_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()



