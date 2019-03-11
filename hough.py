import cv2
import numpy
import math
import matplotlib.pyplot as plt


def hough(img):
    width, height = img.shape
    theta = numpy.deg2rad(numpy.arange(-90.0, 90.0, 1))
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = numpy.linspace(-diag_len, diag_len, diag_len * 2)
    cos_t = numpy.cos(theta)
    sin_t = numpy.sin(theta)
    num_thetas = len(theta)

    accumulator = numpy.zeros((2 * diag_len, num_thetas), dtype=numpy.uint8)
    edges = img > 0
    y_idxs, x_idxs = numpy.nonzero(edges)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    return accumulator, theta, rhos


def show_hough_line(img, accumulator, thetas, rhos):
    plt.imshow(accumulator, cmap='gray', extent=[numpy.rad2deg(thetas[-1]), numpy.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    plt.title('Hough transform')
    plt.xlabel('Angles (degrees)')
    plt.ylabel('Distance (pixels)')
    plt.axis('image')
    plt.show()


image = cv2.imread("lines.jpg")
greyimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurimg = cv2.GaussianBlur(greyimg, (5, 5), 0)
edgeimg = cv2.Canny(blurimg, 50, 150)
cv2.imshow("image", edgeimg)
cv2.waitKey(0)
acc, theta, rho = hough(edgeimg)
show_hough_line(image, acc, theta, rho)
