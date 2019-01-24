import glob
import numpy as np
from PIL import Image
import time

tall = 0
wide = 0

w = 128
h = 128
counter =  0
avg = 0
for image_filename in glob.glob('data/*'):
    #img = np.array(Image.open(image_filename))
    img = Image.open(image_filename)
    #print(img.size)
    #print(image_filename)
    try:
        img = img[:-48,:,:]
    except:
        #print(img.shape)
        #print(image_filename)
        continue
    if img.size[0]/img.size[1] < 2.2 and img.size[1]/img.size[0] < 2.2:
        counter += 1
    if img.shape[0]/img.shape[1] > 1.5:
        tall += 1
    # h/w
    elif img.shape[1]/img.shape[0] > 2:
        wide += 1
        avg += img.shape[1]/img.shape[0]
        #counter += 1
        #Image.fromarray(img, 'RGB').resize((w,h)).show()
        #if counter == 5:
        #    break
        #Image.fromarray(img, 'RGB').show()
#avg = avg/wide

print(counter)

# TODO
# setup and fix gan
#   SAGAN
#   hinge loss?
#   spec norm
# load images into GAN
# decide what size of photos to use

