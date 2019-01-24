import glob
import numpy as np
from PIL import Image
import random

def data_generator(batch_size, img_dir, image_size):
    # Get image filenames and shuffle them
    image_filenames = glob.glob(img_dir + "/*")
    random.shuffle(image_filenames)

    # The images are saved with a watermark at the bottom.
    # Load the image, remove the watermark, resize it into a square,
    # and then put it into a numpy array ready to stick it into the GAN
    file_counter = 0
    while True:
        # If making a new batch would cause index out of bounds error,
        # don't make a new batch and break
        if file_counter + batch_size >= len(image_filenames):
            break

        # batch_counter inc's only when adding an image to the batch
        # file_counter inc's whenever we look at a photo, even if the
        #   size is wrong. To index correctly, only inc file_counter at
        #   seeing wrong sized image, at the end
        batch_of_images = np.zeros((batch_size, image_size, image_size, 2))
        batch_counter = 0
        while batch_counter < batch_size:
            img = Image.open(image_filenames[file_counter+batch_counter]).convert('LA')
            img = img.crop((0,0,img.size[0], img.size[1]-48))
            # The idea here is only use approximately square images
            if img.size[0]/img.size[1] < 2.2 and \
               img.size[1]/img.size[0] < 2.2:
                img = img.resize((image_size, image_size))
                img = np.array(img)
                batch_of_images[batch_counter] = img
                batch_counter += 1
            else:
                file_counter += 1
        file_counter += batch_size
        yield batch_of_images/255

if __name__ == "__main__":
    tracker = 0
    for i, batch in enumerate(data_generator(16, 'data', 64)):
        tracker = i
    print(tracker)
