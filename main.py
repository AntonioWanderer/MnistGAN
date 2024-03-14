import cv2
import numpy
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, BatchNormalization, \
    Reshape, Flatten, LeakyReLU

batch_size = 32
noise_dim = 10
(x, y), (xT, yT) = keras.datasets.mnist.load_data()
digit = 0
xD = []
for i in range(x.shape[0]):
    im = x[i]
    if y[i] == digit:
        xD.append(im)
xD = numpy.array(xD) / 255
xD = numpy.expand_dims(xD, axis=3)

generator = Sequential()
# generator.add(Input(shape=noise_dim))
generator.add(Dense(7 * 7 * 1))
generator.add(LeakyReLU(alpha=0.05))
generator.add(Dense(14 * 14 * 1))
generator.add(LeakyReLU(alpha=0.05))
generator.add(Dense(28 * 28 * 1))
generator.add(LeakyReLU(alpha=0.05))
generator.add(Reshape(target_shape=(28, 28, 1)))
generator.add(Conv2DTranspose(filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same"))
generator.compile(loss="mse")

discriminator = Sequential()
discriminator.add(Input(shape=(28, 28, 1)))
discriminator.add(Conv2D(filters=32, kernel_size=(3, 3)))
discriminator.add(LeakyReLU(alpha=0.05))
discriminator.add(BatchNormalization())
discriminator.add(MaxPooling2D(pool_size=(2, 2)))
discriminator.add(Conv2D(filters=32, kernel_size=(3, 3)))
discriminator.add(LeakyReLU(alpha=0.05))
discriminator.add(BatchNormalization())
discriminator.add(MaxPooling2D(pool_size=(2, 2)))
discriminator.add(Flatten())
# discriminator.add(Dense(800))
# discriminator.add(Dense(600))
discriminator.add(Dense(400))
discriminator.add(LeakyReLU(alpha=0.05))
discriminator.add(BatchNormalization())
discriminator.add(Dense(200))
discriminator.add(LeakyReLU(alpha=0.05))
discriminator.add(BatchNormalization())
discriminator.add(Dense(100))
discriminator.add(LeakyReLU(alpha=0.05))
discriminator.add(BatchNormalization())
discriminator.add(Dense(1, activation="sigmoid"))
discriminator.compile(loss="binary_crossentropy")

gan_input = Input(shape=(noise_dim,))
g = generator(gan_input)
gan_output = discriminator(g)
gan = Model(inputs=gan_input, outputs=gan_output)
gan.compile(loss="binary_crossentropy")

print(generator.summary())
print(discriminator.summary())

truth = numpy.ones(shape=(batch_size, 1))
falth = numpy.zeros(shape=(batch_size, 1))
for epoch in range(1000000):
    image_batch = numpy.array([xD[numpy.random.randint(0, xD.shape[0] - 1)] for _ in range(batch_size)])
    noise = numpy.random.random(size=(batch_size, noise_dim))
    generation = generator.predict(noise, verbose=False)
    discriminator.trainable = True
    discriminator.train_on_batch(image_batch, truth)
    discriminator.train_on_batch(generation, falth)
    discriminator.trainable = False
    gan.train_on_batch(noise, truth)
    if epoch % 10 == 0:
        loss_T = discriminator.evaluate(image_batch, truth, verbose=False)
        loss_F = discriminator.evaluate(generation, falth, verbose=False)
        print(f"true: {loss_T}, false {loss_F}")
        cv2.imwrite(f"results/result{epoch}.jpg", generation[0]*255)
        # generator.save(f"models/generator{epoch}")
        # discriminator.save(f"models/discriminator{epoch}")
