import cv2
import os
import Config
import gc
import numpy
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, BatchNormalization, \
    Reshape, Flatten, LeakyReLU, AveragePooling2D

def get_fashion():
    (x, y), (xT, yT) = keras.datasets.fashion_mnist.load_data()
    digit = Config.digit
    xD = []
    for i in range(x.shape[0]):
        im = x[i]
        if y[i] == digit:
            xD.append(im)
    xD = numpy.array(xD) / Config.divisor
    xD = numpy.expand_dims(xD, axis=3)
    print(xD.shape)
    return xD

def get_mnist():
    (x, y), (xT, yT) = keras.datasets.mnist.load_data()
    digit = Config.digit
    xD = []
    for i in range(x.shape[0]):
        im = x[i]
        if y[i] == digit:
            xD.append(im)
    xD = numpy.array(xD) / Config.divisor
    xD = numpy.expand_dims(xD, axis=3)
    print(xD.shape)
    return xD


def get_images(path=Config.path_content):
    files = os.listdir(path)
    data = []
    for name in files:
        try:
            image = cv2.imread(filename=path + name)
            resized = cv2.resize(src=image, dsize=Config.imShape)
            # gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            data.append(resized)
            # data.append(gray)
        except:
            print("Error")
    data = numpy.array(data)
    print(data.shape)
    return data


def normalization(tensor, reverse=False):
    if reverse:
        return tensor * Config.divisor
    else:
        return tensor / Config.divisor


def get_model():
    generator = Sequential()
    # generator.add(Input(shape=noise_dim))
    generator.add(Dense(10*10*3))
    generator.add(LeakyReLU(alpha=Config.alpha))
    generator.add(Dense(20*20*3))
    generator.add(LeakyReLU(alpha=Config.alpha))
    generator.add(Dense(Config.w * Config.h * Config.d))
    generator.add(LeakyReLU(alpha=Config.alpha))
    generator.add(Reshape(target_shape=(Config.w, Config.h, Config.d)))
    generator.add(Conv2DTranspose(filters=Config.d, kernel_size=(3, 3), activation="sigmoid", padding="same"))
    generator.compile(loss="mse", optimizer="adam")

    discriminator = Sequential()
    discriminator.add(Input(shape=(Config.w, Config.h, Config.d)))
    discriminator.add(Conv2D(filters=32, kernel_size=(3, 3)))
    discriminator.add(LeakyReLU(alpha=Config.alpha))
    discriminator.add(BatchNormalization())
    discriminator.add(AveragePooling2D(pool_size=(2, 2)))
    discriminator.add(Conv2D(filters=32, kernel_size=(3, 3)))
    discriminator.add(LeakyReLU(alpha=Config.alpha))
    discriminator.add(BatchNormalization())
    discriminator.add(AveragePooling2D(pool_size=(2, 2)))
    discriminator.add(Flatten())
    discriminator.add(Dense(50))
    discriminator.add(LeakyReLU(alpha=Config.alpha))
    discriminator.add(Dense(30))
    discriminator.add(LeakyReLU(alpha=Config.alpha))
    discriminator.add(Dense(10))
    discriminator.add(LeakyReLU(alpha=Config.alpha))
    discriminator.add(Dense(1, activation="sigmoid"))
    discriminator.compile(loss="binary_crossentropy", optimizer="adam")

    gan_input = Input(shape=(Config.noise_dim,))
    g = generator(gan_input)
    gan_output = discriminator(g)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss="binary_crossentropy", optimizer="adam")

    print(generator.summary())
    print(discriminator.summary())

    return generator, discriminator, gan


def train(xD, generator, discriminator, gan):
    truth = numpy.ones(shape=(Config.batch_size, 1))
    falth = numpy.zeros(shape=(Config.batch_size, 1))
    for epoch in range(1000000):
        image_batch = numpy.array([xD[numpy.random.randint(0, xD.shape[0] - 1)] for _ in range(Config.batch_size)])
        noise = numpy.random.normal(size=(Config.batch_size, Config.noise_dim))
        generation = generator.predict(noise, verbose=False)
        for j in range(10):
            discriminator.trainable = True
            discriminator.train_on_batch(image_batch, truth)
            discriminator.train_on_batch(generation, falth)
            discriminator.trainable = False
            gan.train_on_batch(noise, truth)
            #generator.train_on_batch(noise,image_batch)
        if epoch % 10 == 0:
            loss_T = discriminator.evaluate(image_batch, truth, verbose=False)
            loss_F = discriminator.evaluate(generation, falth, verbose=False)
            print(f"true: {loss_T}, false {loss_F}")
            cv2.imwrite(f"results/result{epoch}.jpg", generation[0] * Config.divisor)
            generator.save(f"models/generator{epoch}")
            discriminator.save(f"models/discriminator{epoch}")
            del loss_T
            del loss_F
        del generation
        del noise
        del image_batch
        gc.collect()
        keras.backend.clear_session()


if __name__ == "__main__":
    collection = get_fashion()
    # collection = get_mnist()
    # collection = get_images()
    generator, discriminator, gan = get_model()
    train(collection, generator, discriminator, gan)
