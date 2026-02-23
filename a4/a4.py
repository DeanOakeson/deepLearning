import numpy as np
import random
import matplotlib.pyplot as plt
from skimage.transform import iradon
from skimage.util import random_noise
import tensorflow as tf
import skimage.io
import skimage.filters
from tensorflow import keras
from tensorflow.keras import layers
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Input, InputLayer

def phantom(n=256, p_type="modified shepp-logan", ellipses=None):
    # [[I, a, b, x0, y0, phi],
    #  [I, a, b, x0, y0, phi]]
    # I : Additive intensity of the ellipse.
    # a : Length of the major axis.
    # b : Length of the minor axis.
    # x0 : Horizontal offset of the centre of the ellipse.
    # y0 : Vertical offset of the centre of the ellipse.
    # phi : Counterclockwise rotation of the ellipse in degrees,
    #       measured as the angle between the horizontal axis and
    #       the ellipse major axis.
    # The image bounding box in the algorithm is [-1, -1], [1, 1],
    # so the values of a, b, x0, y0 should all be specified with
    # respect to this box
    # outpu P : a phantom image.

    if ellipses is None:
        ellipses = _select_phantom(p_type)
    elif np.size(ellipses, 1) != 6:
        raise AssertionError("Wrong number of columns in user phantom")

    p = np.zeros((n, n))

    ygrid, xgrid = np.mgrid[-1 : 1 : (1j * n), -1 : 1 : (1j * n)]

    for ellip in ellipses:
        I = ellip[0]
        a2 = ellip[1] ** 2
        b2 = ellip[2] ** 2
        x0 = ellip[3]
        y0 = ellip[4]
        phi = ellip[5] * np.pi / 180

        x = xgrid - x0
        y = ygrid - y0

        cos_p = np.cos(phi)
        sin_p = np.sin(phi)

        # find the pixels within the ellipse
        locs = (
            ((x * cos_p + y * sin_p) ** 2) / a2 + ((y * cos_p - x * sin_p) ** 2) / b2
        ) <= 1

        # add the ellipse intensity to those pixels
        p[locs] += I

    return p


def _select_phantom(name):
    if name.lower() == "shepp-logan":
        e = _shepp_logan()
    elif name.lower() == "modified shepp-logan":
        e = _mod_shepp_logan()
    else:
        raise ValueError("Unknown phantom type: %s" % name)
    return e


def _shepp_logan():
    # standard head phantom, taken from shepp and logan

    return [
        [2, 0.69, 0.92, 0, 0, 0],
        [-0.98, 0.6624, 0.8740, 0, -0.0184, 0],
        [-0.02, 0.1100, 0.3100, 0.22, 0, -18],
        [-0.02, 0.1600, 0.4100, -0.22, 0, 18],
        [0.01, 0.2100, 0.2500, 0, 0.35, 0],
        [0.01, 0.0460, 0.0460, 0, 0.1, 0],
        [0.02, 0.0460, 0.0460, 0, -0.1, 0],
        [0.01, 0.0460, 0.0230, -0.08, -0.605, 0],
        [0.01, 0.0230, 0.0230, 0, -0.606, 0],
        [0.01, 0.0230, 0.0460, 0.06, -0.605, 0],
    ]


def _mod_shepp_logan():
    #  Modified version of Shepp & Logan's head phantom,
    #  adjusted to improve contrast.  Taken from Toft.
    return [
        [1, 0.69, 0.92, 0, 0, 0],
        [-0.80, 0.6624, 0.8740, 0, -0.0184, 0],
        [-0.20, 0.1100, 0.3100, 0.22, 0, -18],
        [-0.20, 0.1600, 0.4100, -0.22, 0, 18],
        [0.10, 0.2100, 0.2500, 0, 0.35, 0],
        [0.10, 0.0460, 0.0460, 0, 0.1, 0],
        [0.10, 0.0460, 0.0460, 0, -0.1, 0],
        [0.10, 0.0460, 0.0230, -0.08, -0.605, 0],
        [0.10, 0.0230, 0.0230, 0, -0.606, 0],
        [0.10, 0.0230, 0.0460, 0.06, -0.605, 0],
    ]


def main():

    ##CNN BASE##
    # model = keras.Sequential([
    #     Conv2D(2, (3,3), activation='relu', padding='same', strides=1, use_bias=False),
    #     Conv2D(2, (3,3), activation='relu', padding='same', strides=1, use_bias=False),
    #     Conv2D(2, (3,3), activation='relu', padding='same', strides=1, use_bias=False),
    #     Conv2D(1, (3,3), activation='relu', padding='same', use_bias=False)
    # ])
    
    ##CNN WORSE##
    # model = keras.Sequential([
    #     Conv2D(2, (5,5), activation='relu', padding='same', strides=1, use_bias=False),
    #     Conv2D(1, (3,3), activation='relu', padding='same', use_bias=False)
    # ])
    
    ##CNN BETTER##
    # model = keras.Sequential([
    #     Conv2D(4, (3,3), activation='relu', padding='same', strides=1, use_bias=False),
    #     Conv2D(4, (5,5), activation='relu', padding='same', strides=1, use_bias=False),
    #     Conv2D(4, (7,7), activation='relu', padding='same', strides=1, use_bias=False),
    #     Conv2D(4, (5,5), activation='relu', padding='same', strides=1, use_bias=False),
    #     Conv2D(4, (3,3), activation='relu', padding='same', strides=1, use_bias=False),
    #     Conv2D(1, (3,3), activation='linear', padding='same', use_bias=False)
    #
    #        ])

    ## UNET BASE##
    # model = keras.Sequential([
    #  layers.Conv2D(15, 3, strides=2, activation="relu", padding="same"),
    #  layers.Conv2D(15, 3, activation="relu", padding="same"),
    #  layers.Conv2D(20, 3, strides=2, activation="relu", padding="same"),
    #  layers.Conv2D(20, 3, activation="relu", padding="same"),
    #  layers.Conv2D(25, 3, strides=2, padding="same", activation="relu"),
    #  layers.Conv2D(25, 3, activation="relu", padding="same"),
    #  layers.Conv2DTranspose(25, 3, activation="relu", padding="same"),
    #  layers.Conv2DTranspose(25, 3, activation="relu", padding="same", strides=2),
    #  layers.Conv2DTranspose(20, 3, activation="relu", padding="same"),
    #  layers.Conv2DTranspose(20, 3, activation="relu", padding="same", strides=2),
    #  layers.Conv2DTranspose(15, 3, activation="relu", padding="same"),
    #  layers.Conv2DTranspose(15, 3, activation="relu", padding="same", strides=2),
    #  layers.Conv2D(1, 3, activation="relu", padding="same")
    # ])

    ## UNET WORSE ##
    # model = keras.Sequential([
    #  layers.Conv2D(15, 3, strides=2, activation="relu", padding="same"),
    #  layers.Conv2D(15, 3, activation="relu", padding="same"),
    #  layers.Conv2D(20, 3, strides=2, activation="relu", padding="same"),
    #  layers.Conv2D(20, 3, activation="relu", padding="same"),
    #  layers.Conv2D(25, 3, strides=2, padding="same", activation="relu"),
    #  layers.Conv2D(25, 3, activation="relu", padding="same"),
    #  layers.Conv2D(30, 3, strides=2, padding="same", activation="relu"),
    #  layers.Conv2D(30, 3, activation="relu", padding="same"),
    #  layers.Conv2D(35, 3, strides=2, padding="same", activation="relu"),
    #  layers.Conv2D(35, 3, activation="relu", padding="same"),
    #
    #
    #  layers.Conv2DTranspose(35, 3, activation="relu", padding="same"),
    #  layers.Conv2DTranspose(35, 3, activation="relu", padding="same", strides=2),
    #  layers.Conv2DTranspose(30, 3, activation="relu", padding="same"),
    #  layers.Conv2DTranspose(30, 3, activation="relu", padding="same", strides=2),
    #  layers.Conv2DTranspose(25, 3, activation="relu", padding="same"),
    #  layers.Conv2DTranspose(25, 3, activation="relu", padding="same", strides=2),
    #  layers.Conv2DTranspose(20, 3, activation="relu", padding="same"),
    #  layers.Conv2DTranspose(20, 3, activation="relu", padding="same", strides=2),
    #  layers.Conv2DTranspose(15, 3, activation="relu", padding="same"),
    #  layers.Conv2DTranspose(15, 3, activation="relu", padding="same", strides=2),
    #  layers.Conv2D(1, 3, activation="relu", padding="same")
    # ])

    ## UNET BETTER ##
    model = keras.Sequential([
     layers.Conv2D(20, 3, strides=2, activation="relu", padding="same"),
     layers.Conv2D(20, 3, activation="relu", padding="same"),
     layers.Conv2D(40, 3, strides=2, activation="relu", padding="same"),
     layers.Conv2D(40, 3, activation="relu", padding="same"),
     layers.Conv2DTranspose(40, 3, activation="relu", padding="same"),
     layers.Conv2DTranspose(40, 3, activation="relu", padding="same", strides=2),
     layers.Conv2DTranspose(20, 3, activation="relu", padding="same"),
     layers.Conv2DTranspose(20, 3, activation="relu", padding="same", strides=2),
     layers.Conv2D(1, 3, activation="linear", padding="same")
    ])

    x = np.random.random(24)
    print(x[5])

# generate data
    image_size = 64
    num_phantoms = 1000

    noise_factor = 0.05     #0.1
    noisy_img = []
    clean_img = []
    noisy_imgV = []
    clean_imgV = []
    clean_phan = []

    img_size1 = 64
    img_size2 = 64

    for i in range(num_phantoms):
        
        a = np.random.randint(20,30,2)
        #print(a)
        
        
        x=np.random.random(24)
        
        E = [[   x[ 0]-0.0,   0.6*x[ 1]+0.2,   0.8*x[ 2]+0.2,    0.1*(x[ 3]-0.5),      0.1*(x[ 4]-0.5),   10*x[ 5]   ],
             [   x[ 6]-0.1,   0.3*x[ 7]+0.2,   0.2*x[ 8]+0.2,    x[ 9]-0.5,      x[10]-0.5,   100*x[11]   ],
             [   x[12]-0.2,   0.2*x[13]+0.2,   0.3*x[14]+0.2,    x[15]-0.5,      x[16]-0.5,   100*x[17]   ],
             [   x[18]-0.3,   0.1*x[19]+0.2,   0.1*x[20]+0.2,    x[21]-0.5,      x[22]-0.5,   100*x[23]  ]] 
        P = phantom (n = image_size, p_type = 'ellipses', ellipses = E)
        
        Pmax = np.max(P )
        P0 =  P/(2*Pmax)
        P0 = np.maximum(0, P0)
        P0 = np.minimum(0.5, P0)
        
        P1  = P0 + noise_factor * tf.random.normal(shape=P0.shape)
        
        
        # P1 = random_noise(P/1000, mode='poisson', seed=None, clip=True)
        P1 = np.maximum(0, P1)
        P1 = np.minimum(1, P1)
        
        clean_phan.append(P)    # Phantomss
        
        if i < 0.9*num_phantoms:
            noisy_img.append(P1)    # Data
            clean_img.append(P0)    # Targets
        else:
            noisy_imgV.append(P1)    # Data
            clean_imgV.append(P0)    # Targets
        
    np_noisy_img = np.asarray(noisy_img)
    np_clean_img = np.asarray(clean_img)
    np_noisy_imgV = np.asarray(noisy_imgV)
    np_clean_imgV = np.asarray(clean_imgV)
    np_clean_phan = np.asarray(clean_phan)      

    # prepare data axes as expected by models
    np_noisy = np.expand_dims(np_noisy_img, axis=-1)
    np_clean = np.expand_dims(np_clean_img, axis=-1)
    np_noisyV = np.expand_dims(np_noisy_imgV, axis=-1)
    np_cleanV = np.expand_dims(np_clean_imgV, axis=-1)
    np_phan = np.expand_dims(np_clean_phan, axis=-1)
    print(np.shape(np_noisy))

    print(np.max(np_noisy))
    print(np.max(np_clean))


    model.compile(optimizer='adam', loss='mse')

    #optimizer = keras.optimizers.Adam(lr=0.01)
    #model.compile(loss='mse', optimizer=optimizer)

    model.fit(np_noisy, 
              np_clean, 
              epochs=100, 
              shuffle=True, 
              validation_data=(np_noisyV, np_cleanV))

    decoded_imgs=model(np_noisyV).numpy()

    n = 10 
    plt.figure(figsize=(20, 7))
    plt.title("parameters used: %i" %model.count_params())
    plt.gray()
    for i in range(n): 
      # display original + noise 
      bx = plt.subplot(3, n, i + 1) 
      plt.title("original + noise") 
      plt.imshow(tf.squeeze(np_noisyV[i])) 
      bx.get_xaxis().set_visible(False) 
      bx.get_yaxis().set_visible(False) 
      
      # display reconstruction 
      cx = plt.subplot(3, n, i + n + 1) 
      plt.title("reconstructed") 
      plt.imshow(tf.squeeze(decoded_imgs[i])) 
      cx.get_xaxis().set_visible(False) 
      cx.get_yaxis().set_visible(False) 
      
      # display original 
      ax = plt.subplot(3, n, i + 2*n + 1) 
      plt.title("original") 
      plt.imshow(tf.squeeze(np_cleanV[i])) 
      ax.get_xaxis().set_visible(False) 
      ax.get_yaxis().set_visible(False) 

    plt.show()


   
main()
