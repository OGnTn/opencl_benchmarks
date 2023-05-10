#!/usr/bin/env python3

import os
import sys
import time
import pyopencl as cl
import numpy
from libs.util import *

# Parse CLI arguments
IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else "input/img0.jpg"
KERNEL_NAME = sys.argv[2] if len(sys.argv) > 2 else "kernel_naive"
# Work group size will be (LOCAL_SIZE, LOCAL_SIZE).
LOCAL_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 16

# Kernel dimensions, default to 5.
KERNEL_DIM = int(sys.argv[4]) if len(sys.argv) > 4 else 5
# Index of the middle of the kernel, e.g. 2 for a 5x5 kernel.
kernel_mid = KERNEL_DIM // 2
# Sigma used for Gaussian kernel.
KERNEL_SIGMA = 0.84089642
print("Kernel dimensions:", KERNEL_DIM)
print("Sigma:", KERNEL_SIGMA)
# A 5x5 kernel with sigma 0.84089642.
convolution_kernel = gaussian_kernel(KERNEL_DIM, KERNEL_SIGMA)
print("Kernel:", convolution_kernel)

N_ITERATIONS = 30

# Suppress kernel caching.
os.environ["PYOPENCL_NO_CACHE"] = "1"

# Load input image and add alpha channel.
img_no_padding = image_to_array(IMG_PATH)
img_no_padding = add_alpha_channel(img_no_padding)

# Add padding with white pixels to the image.
img = pad_image(img_no_padding, kernel_mid, [255, 255, 255, 0])

# Determine height and width of both original and padded image.
(img_h, img_w, depth) = img.shape
(img_original_h, img_original_w, _) = img_no_padding.shape
print(
    f"Image dimensions: {img_original_h}x{img_original_w} with depth {depth}")
print(f"Image dimensions after padding: {img_h}x{img_w}")

# Flatten the image and the kernel, and make sure the types are correct.
flat_img = img.reshape((img_h * img_w * depth)).astype(numpy.uint8)
flat_kernel = convolution_kernel.reshape((KERNEL_DIM * KERNEL_DIM)).astype(
    numpy.float32
)
# Create the result image.
h_output_img = numpy.zeros(img_h * img_w * depth).astype(numpy.uint8)

# Create the context, queue and program.
context = cl.create_some_context()
queue = cl.CommandQueue(context)
kernel_code = open(KERNEL_NAME + ".cl").read()
program = cl.Program(context, kernel_code).build()

# Create the buffers on the device.
d_input_img = cl.Buffer(
    context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=flat_img
)
d_kernel = cl.Buffer(
    context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=flat_kernel
)
d_output_img = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_output_img.nbytes)

# Initialize the kernel.
conv = program.convolve
conv.set_scalar_arg_dtypes(
    [None, None, None, numpy.int32, numpy.int32,
        numpy.int32, numpy.int32, numpy.int32]
)

# Execute.
for i in range(N_ITERATIONS):
    print()
    print("Iteration", i)

    start_time = time.perf_counter()
    conv(
        queue,
        (img_original_h, img_original_w),
        (LOCAL_SIZE, LOCAL_SIZE),
        d_input_img,
        d_output_img,
        d_kernel,
        KERNEL_DIM,
        kernel_mid,
        img_w,
        img_h,
        depth,
        global_offset=[kernel_mid, kernel_mid],
    )
    queue.finish()
    end_time = time.perf_counter()

    exec_time = end_time - start_time
    print("Time to execute kernel: %.4f seconds" % exec_time)

# Read the array from the device.
cl.enqueue_copy(queue, h_output_img, d_output_img)

# Reshape the image array
result_img = h_output_img.reshape(img.shape)

# Now remove the padding from image
result_img_no_padding = result_img[kernel_mid:-
                                   kernel_mid, kernel_mid:-kernel_mid, :]

# Final image
image = numpy.asarray(result_img_no_padding).astype(dtype=numpy.uint8)
# verify_image("../output/output_seq_img1.bmp", image, kernel_mid)
save_image_rgba(image, "output_" + KERNEL_NAME + ".bmp")
