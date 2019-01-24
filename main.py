import glob, random, time
import numpy as np
import keras.backend as K
from scipy import misc
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, Dense, BatchNormalization, LeakyReLU, Input, Flatten, Reshape, Conv2DTranspose, GlobalAveragePooling2D
from keras.optimizers import Adam, SGD
from snconv import *
from data_loader import data_generator


def one_photo_data_generator(batch_size, img_dir, img_size):
    w = img_size
    h = img_size
    image_filename = img_dir + "/overfit_image.jpg"
    counter = 0
    while True:
        img_data = np.zeros((batch_size, w, h, 3))
        if ((counter+batch_size)>=100):
            break
        for i in range(batch_size):
            img = Image.open(image_filename).resize((w, h))
            try:
                img_data[i] = np.array(img)
            except:
                print('failed on image:', image_filenames[counter+i])
                print(np.array(img).shape)
        yield img_data/255

def get_generator(input_shape):
    generator = Sequential()

    generator.add(Conv2DTranspose(512, (3,3), strides=(2,2), padding="same", input_shape=input_shape))
    generator.add(LeakyReLU(0.2))

    generator.add(Conv2DTranspose(256, (3,3), strides=(2,2), padding="same"))
    generator.add(LeakyReLU(0.2))

    generator.add(Conv2DTranspose(128, (3,3), strides=(2,2), padding="same"))
    generator.add(LeakyReLU(0.2))

    generator.add(Conv2DTranspose(64, (3,3), strides=(2,2), padding="same"))
    generator.add(LeakyReLU(0.2))

    generator.add(Conv2DTranspose(32, (3,3), strides=(2,2), padding="same"))
    generator.add(LeakyReLU(0.2))

    generator.add(Conv2DTranspose(16, (3,3), padding="same"))
    generator.add(LeakyReLU(0.2))

    generator.add(Conv2DTranspose(16, (3,3), padding="same"))
    generator.add(LeakyReLU(0.2))

    generator.add(Conv2DTranspose(16, (3,3), padding="same"))
    generator.add(LeakyReLU(0.2))

    generator.add(Conv2DTranspose(2, (2,2), padding="same", activation="tanh"))
    return generator

def get_discriminator(input_shape):
    discriminator = Sequential()
    discriminator.add(Conv2D(64, (2,2), strides=(2,2), padding='same', input_shape=(64,64,2)))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(64, (3,3), strides=(2,2), padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(64, (3,3), strides=(2,2), padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(512, (3,3), strides=(2,2), padding='same'))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(1, (3,3), strides=(2,2), padding='same', activation="sigmoid"))
    #discriminator.add(GlobalAveragePooling2D())

    return discriminator

def get_discriminator_SN(input_shape):
    discriminator = Sequential()
    discriminator.add(SNConv2D(64,(2,2),strides=(2,2),padding="same",input_shape=input_shape))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(SNConv2D(64,(3,3),strides=(2,2),padding="same"))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(SNConv2D(64,(3,3),strides=(2,2),padding="same"))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(SNConv2D(128,(3,3),strides=(2,2),padding="same"))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(SNConv2D(256,(3,3),strides=(2,2),padding="same"))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(SNConv2D(512,(3,3),strides=(2,2),padding="same"))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(SNConv2D(1,(3,3),padding="same"))
    #discriminator.add(Conv2D(1, (3,3), strides=(2,2), padding='same', activation="sigmoid"))
    discriminator.add(GlobalAveragePooling2D())

    return discriminator


def define_training_functions(gen, dis):

    real_img = Input((image_size, image_size, 2))
    noise = K.random_normal((batch_size,) + noise_dim, 0.0, 1.0, "float32")
    fake_img = gen(noise)
    pred_real = dis(real_img)
    pred_fake = dis(fake_img)

    d_loss = -K.mean(K.minimum(0., -1 + pred_real)) - K.mean(K.minimum(0., -1 - pred_fake))
    g_loss = -K.mean(pred_fake)

    d_training_updates = Adam(lr=0.0001,beta_1=0.0,beta_2=0.9).get_updates(d_loss, dis.trainable_weights)
    d_train = K.function([real_img, K.learning_phase()], [d_loss], d_training_updates)

    g_training_updates = Adam(lr=0.0001,beta_1=0.0,beta_2=0.9).get_updates(g_loss, gen.trainable_weights)
    g_train = K.function([K.learning_phase()], [g_loss], g_training_updates)

    return d_train, g_train


def predict_images(g, n=3, filename='clouds.png'):
    image = np.zeros(shape=(image_size*n, image_size*n, 2))
    for i in range(0, image_size*n, image_size):
        for j in range(0, image_size*n, image_size):
            image[i:i+image_size, j:j+image_size, :] = g.predict(np.random.normal(size=(1,) + noise_dim))[0]
    image = image*255
    image = image.astype('uint8')
    Image.fromarray(image).save("outputs/"+filename)
    #imisc.imsave('outputs/'+filename, image)

def train(epochs, g, d, print_every=100):
    d_train, g_train = define_training_functions(g, d)
    for epoch in range(epochs):
        print("Epoch #", epoch+1)
        epoch_start = time.time()
        data_gen = data_generator(batch_size, 'data', image_size)
        for i, img_batch in enumerate(data_gen):
            d_loss, = d_train([img_batch, 1])
            g_loss = g_train([1])

            if i % print_every == 0:
                print("Batch %d d_loss: %f g_loss: %f" % (i, d_loss, g_loss[0]))
        predict_images(g, 3, "bw_epoch_%d.png" % epoch )
        epoch_end = time.time()-epoch_start/60
        print("End of Epoch {}. Time Elapsed: {}".format(epoch, epoch_end))




image_size = 64
batch_size = 32
noise_dim = (2, 2, 256)

g = get_generator(noise_dim)
d = get_discriminator_SN((image_size, image_size, 2))
g.summary()
d.summary()
train(1000, g, d)


# TODO
# train D more than G
# put SN in G
