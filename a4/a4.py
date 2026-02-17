import matplotlib.pyplot as plt
import numpy as np
import skimage


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
    print("hello world\n")
    print(plt.rcParams)

    x = np.random.random(24)
    E = [
        [
            x[0] - 0.0,
            0.5 * x[1] + 0.2,
            0.5 * x[2] + 0.2,
            x[3] - 0.5,
            x[4] - 0.5,
            100 * x[5],
        ],
        [
            x[6] - 0.1,
            0.4 * x[7] + 0.2,
            0.4 * x[8] + 0.2,
            x[9] - 0.5,
            x[10] - 0.5,
            100 * x[11],
        ],
        [
            x[12] - 0.2,
            0.3 * x[13] + 0.2,
            0.3 * x[14] + 0.2,
            x[15] - 0.5,
            x[16] - 0.5,
            100 * x[17],
        ],
        [
            x[18] - 0.3,
            0.2 * x[19] + 0.2,
            0.2 * x[20] + 0.2,
            x[21] - 0.5,
            x[22] - 0.5,
            100 * x[23],
        ],
    ]

    P = phantom(n=256, p_type="ellipses", ellipses=E)
    # pl.imshow (P, cmap=plt.cm.Greys_r)

    sigma = 4
    blurred = skimage.filters.gaussian(P, sigma=(sigma, sigma), truncate=3.5)
    # skimage.io.imshow(blurred, cmap=plt.cm.Greys_r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5), sharex=True, sharey=True)
    ax1.set_title("Original Shepp-Logan Phantom")
    ax1.imshow(P, cmap=plt.cm.Greys_r)
    plt.show()
    ax2.set_title("Blurred Image")
    ax2.imshow(blurred, cmap=plt.cm.Greys_r)
    plt.show()

    # from a3.py
    # plt.scatter(inputs[:, 0], inputs[:, 1])
    # plt.plot(x, y, "-r")
    # plt.ylim(-0.25, 1.25)
    # plt.xlim(-0.25, 1.25)
    # plt.show()
    #


main()
