import numpy
import math
import cv2
from PIL import Image

#################
# Image helpers #
#################


def verify_image(correct_filename, img, kernel_mid):
    """Verify the image in `img` with the image in `correct_filename`."""
    (h, w, d) = img.shape
    correct = 0
    wrong = 0
    # Note that you may need to increase tolerance when exporting to JPG.
    tolerance = 0
    seq_image = add_alpha_channel(image_to_array(correct_filename)).astype(numpy.int32)
    img = img.astype(numpy.int32)
    # We need to convert to signed integers, because the subtraction below can
    # result in negative values.
    for i in range(kernel_mid, (h - kernel_mid)):
        for j in range(kernel_mid, (w - kernel_mid)):
            seq_pixel = seq_image[i][j]
            other_pixel = img[i][j]
            # Use the red byte to check for consistency
            error = numpy.absolute(seq_pixel[0] - other_pixel[0])
            if error <= tolerance:
                correct += 1
            else:
                print(f"Error at {i}, {j}: expected {seq_pixel[0]}, got {other_pixel[0]}, {error}")
                wrong += 1
    print("correct == {}, wrong = {}".format(correct, wrong))
    return correct / (correct + wrong)


def pad_image(arr, kernel_mid, value):
    """Extend the image in arr with `kernel_mid` pixels on each side, which are
    set to `value`."""
    (height, width, depth) = arr.shape
    # Create new, slightly larger, array
    extended_arr = numpy.empty((height + 2 * kernel_mid, width + 2 * kernel_mid, depth))
    # Fill the (top, bottom, left, right) edges with the value.
    extended_arr[:kernel_mid, :, :] = value
    extended_arr[-kernel_mid:, :, :] = value
    extended_arr[:, :kernel_mid, :] = value
    extended_arr[:, -kernel_mid:, :] = value
    # Copy the original image into the middle of the new array.
    extended_arr[
        kernel_mid : height + kernel_mid, kernel_mid : width + kernel_mid
    ] = arr
    return extended_arr


def add_alpha_channel(img):
    # Split the image to its channels.
    # Note that OpenCV uses BGR, but this is merged again correctly at the end.
    b_channel, g_channel, r_channel = cv2.split(img)
    # Creating a dummy alpha channel.
    alpha_channel = numpy.zeros(b_channel.shape, dtype=b_channel.dtype)
    # Merge the channels back to an image.
    img_with_alpha = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_with_alpha


def save_image_rgb(arr, filename):
    """Takes a numpy array, in the shape (x, y, rgb), and writes it to an image."""
    img = Image.fromarray(arr, "RGB")
    img.save(filename)

def save_image_grey(arr, filename):
    """Takes a numpy array, in the shape (x, y), and writes it to an image."""
    img = Image.fromarray(arr, "L")
    img.save(filename)


def save_image_rgba(arr, filename):
    """Takes a numpy array, in the shape (x, y, rgba), and writes it to an image."""
    img = Image.fromarray(arr, "RGBA").convert("RGB")
    img.save(filename)


def image_to_array(file):
    """Read an image from a file into a numpy 3D array."""
    img = Image.open(file)
    return numpy.asarray(img).astype(numpy.uint8)


##################
# Kernel Helpers #
##################


def normalize_kernel(kernel, dim):
    """Normalizes a kernel, i.e. the sum of all elements is 1."""
    for x in range(0, dim):
        for y in range(dim):
            kernel[x][y] = kernel[x][y] / numpy.sum(kernel)
    return kernel


def gaussian_kernel(dim, sigma):
    """
    The Gaussian blur function is as follows:

                           x² + y²
    G(x,y) =    1        - -------
            --------- * e    2σ²
              2πσ²

    Then, the kernel is normalized to avoid too dark or too light areas.
    """
    rows = dim
    cols = dim
    arr = numpy.empty([rows, cols]).astype(numpy.float32)
    center = dim / 2
    total = 0.0
    for x in range(0, rows):
        for y in range(0, cols):
            x_ = x - center
            y_ = y - center
            arr[x][y] = (1 / (2.0 * math.pi * math.pow(sigma, 2))) * math.pow(
                math.e,
                -1.0
                * ((math.pow(x_, 2) + math.pow(y_, 2)) / (2.0 * math.pow(sigma, 2))),
            )
            total = total + arr[x][y]
    return normalize_kernel(arr, dim)


def identity_kernel(dim):
    """The identity kernel
    0 0 0
    0 1 0
    0 0 0
    """
    arr = numpy.empty([dim, dim]).astype(numpy.float32)
    arr.fill(0.0)
    arr[dim // 2][dim // 2] = 1.0
    return normalize_kernel(
        numpy.array([[-1.0, 0.0, 1.0], [-2.0, 0, 2.0], [-1.0, 0.0, -1.0]]), 3
    )


def blur_kernel():
    """Blurring kernel
    1 2 1
    2 4 2 * 1/16
    1 2 1
    """
    arr = (numpy.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])) * 1 / 16
    return normalize_kernel(arr, 3)
