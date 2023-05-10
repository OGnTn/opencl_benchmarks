#!/usr/bin/env python3
import os
import sys
import time
import pyopencl as cl
import numpy
from PIL import Image, ImageOps
from libs.util import *
import pandas as pd


def pad_image_extra(arr, extra_height, extra_width):
    (height, width, depth) = arr.shape
    # Create new, slightly larger, array
    extended_arr = numpy.zeros(
        (height + extra_height, width + extra_width, depth))
    # Copy the original image into the middle of the new array.
    extended_arr[
        0:height,  0:width
    ] = arr
    return extended_arr


def reshape_and_pad(arr, extra, img_w, img_h):
    arr = arr.reshape((img_h, img_w))
    extended_arr = numpy.zeros((img_h + 2 * extra, img_w + 2 * extra))
    extended_arr[extra:img_h + extra, extra:img_w + extra] = arr
    return extended_arr


# Suppress kernel caching.
#os.environ["PYOPENCL_NO_CACHE"] = "1"


KERNEL_NAME_COMBO = "kernel_combo"
KERNEL_NAME_STARS = "kernel_stars"
kernel_combo_code = open("kernels/" + KERNEL_NAME_COMBO + ".cl").read()
kernel_stars_code = open(
    "kernels/" + KERNEL_NAME_STARS + "_groups" ".cl").read()


def set_context(device_id, image_path):
    global context
    global queue
    global kernel_combo
    global kernel_stars
    global program_combo
    global program_stars
    global IMG_PATH
    global img
    global img_h
    global img_w
    global depth
    global flat_img

    os.environ["PYOPENCL_CTX"] = device_id

    # Create the context, queue and program.
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    program_combo = cl.Program(context, kernel_combo_code).build()
    program_stars = cl.Program(context, kernel_stars_code).build()
    # Initialize the kernel.
    kernel_combo = program_combo.kernel_combo
    kernel_stars = program_stars.kernel_stars
    IMG_PATH = image_path
    img = image_to_array(IMG_PATH)
    (img_h, img_w, depth) = img.shape
    print(f"Image dimensions: {img_h}x{img_w} with depth {depth}")
    flat_img = img.reshape(img_h * img_w * depth).astype(numpy.uint8)


def run_optimized(local_size):

    global img
    global img_h
    global img_w
    global depth
    global flat_img

    #reshape_and_pad(img, 3, img_w, img_h)
    #img_h = img_h + 6
    #img_w = img_w + 6
    if img_h % local_size != 0 or img_w % local_size != 0:
        # Calculate padding required for each dimension
        pad_height = (local_size - img_h %
                      local_size) if img_h % local_size != 0 else 0
        pad_width = (local_size - img_w %
                     local_size) if img_w % local_size != 0 else 0
        img = pad_image_extra(img, pad_height, pad_width)
        (img_h, img_w, depth) = img.shape
        flat_img = img.reshape((img_h * img_w * depth)).astype(numpy.uint8)
        print(f"NEW Image dimensions: {img_h}x{img_w}x{depth}")

    benchmark_dataframe = pd.DataFrame()
    start_total = time.time()

    start_setup_combo = time.time()
    #os.environ["PYOPENCL_CTX"] = device_id

    LOCAL_SIZE_X = local_size
    LOCAL_SIZE_Y = local_size
    # set argument types
    kernel_combo.set_scalar_arg_dtypes(
        [None, None, None,
            numpy.int32, numpy.int32, numpy.int32]
    )

    # Create the result image.
    h_output_img = numpy.zeros(img_h * img_w * 1).astype(numpy.uint8)
    h_output_brightness = numpy.zeros(math.floor(img_w)).astype(numpy.uint32)

    # Create the buffers on the device.
    d_input_img = cl.Buffer(
        context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=flat_img
    )

    d_output_img = cl.Buffer(
        context, cl.mem_flags.WRITE_ONLY, h_output_img.nbytes)
    d_output_brightness = cl.Buffer(
        context, cl.mem_flags.WRITE_ONLY, h_output_brightness.nbytes)

    end_setup_combo = time.time()
    benchmark_dataframe["setup_combo"] = [end_setup_combo-start_setup_combo]

    #print("Executing kernel combo...")
    start_kernel_combo = time.time()
    kernel_combo(
        queue,
        (img_h, img_w),
        (LOCAL_SIZE_X, LOCAL_SIZE_Y),
        d_input_img,
        d_output_img,
        d_output_brightness,
        img_w,
        img_h,
        depth,
    )

    #print("Done executing kernel combo.")

    queue.finish()
    cl.enqueue_copy(queue, h_output_img, d_output_img)
    cl.enqueue_copy(queue, h_output_brightness, d_output_brightness)
    queue.flush()

    threshold = 2 * h_output_brightness.sum()/(img_w * img_h)
    threshold = math.floor(threshold)
    end_kernel_combo = time.time()
    benchmark_dataframe["combo"] = [end_kernel_combo-start_kernel_combo]

    start_setup_stars = time.time()

    LOCAL_SIZE = local_size
    global_size = int((img_w * img_h)/LOCAL_SIZE)

    kernel_stars.set_scalar_arg_dtypes(
        [None, None,
            numpy.int32, numpy.int32, numpy.int32]
    )

    d_input_img = cl.Buffer(
        context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_output_img
    )

    h_input_count = numpy.zeros(
        global_size).astype(numpy.uint32)
    d_output_count = cl.Buffer(
        context, cl.mem_flags.READ_WRITE, h_input_count.nbytes)
    end_setup_stars = time.time()
    benchmark_dataframe["setup_stars"] = [end_setup_stars-start_setup_stars]

    #print("Executing stars kernel...")
    start_stars = time.time()
    kernel_stars(
        queue,
        (global_size, 1),
        (LOCAL_SIZE, 1),
        d_input_img,
        d_output_count,
        threshold,
        img_w,
        img_h
    )

    #print("Done executing stars kernel.")
    queue.finish()
    cl.enqueue_copy(queue, h_input_count, d_output_count)
    queue.flush()
    count = sum(h_input_count)/255
    end_stars = time.time()
    benchmark_dataframe["stars"] = [end_stars-start_stars]
    end_total = time.time()
    benchmark_dataframe["total"] = [end_total-start_total]
    benchmark_dataframe["threshold"] = [threshold]
    benchmark_dataframe["count"] = [count]
    return benchmark_dataframe
    #print("Count: ", sum(h_input_count)/255)
