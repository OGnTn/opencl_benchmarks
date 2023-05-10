import os
import pyopencl as cl
import numpy
from libs.util import *
import pandas as pd
import time


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
#os.environ["PYOPENCL_CTX"] = "0"
KERNEL_NAME_GREY = "kernel_grey"
KERNEL_NAME_THRESHOLD = "kernel_threshold"
KERNEL_NAME_STARS = "kernel_stars"

kernel_grey_code = open("kernels/" + KERNEL_NAME_GREY + ".cl").read()
kernel_threshold_code = open(
    "kernels/" + KERNEL_NAME_THRESHOLD + ".cl").read()
kernel_stars_code = open(
    "kernels/" + KERNEL_NAME_STARS + "_groups" ".cl").read()


def set_context(device_id, image_path):
    print("Setting context...")
    global context
    global queue
    global kernel_grey
    global kernel_threshold
    global kernel_stars
    global program_grey
    global program_threshold
    global program_stars
    global IMG_PATH
    global img
    global img_h
    global img_w
    global depth
    global flat_img

    os.environ["PYOPENCL_CTX"] = device_id
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    program_grey = cl.Program(context, kernel_grey_code).build()
    program_threshold = cl.Program(context, kernel_threshold_code).build()
    program_stars = cl.Program(context, kernel_stars_code).build()
    # Initialize the kernel.
    kernel_grey = program_grey.kernel_grey
    kernel_threshold = program_threshold.kernel_threshold
    kernel_stars = program_stars.kernel_stars

    IMG_PATH = image_path
    img = image_to_array(IMG_PATH)
    (img_h, img_w, depth) = img.shape
    print(f"Image dimensions: {img_h}x{img_w} with depth {depth}")
    flat_img = img.reshape(img_h * img_w * depth).astype(numpy.uint8)


def run_naive(local_size):

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

    start_setup_grey = time.time()
    # set argument types
    LOCAL_SIZE_X = local_size
    LOCAL_SIZE_Y = local_size

    kernel_grey.set_scalar_arg_dtypes(
        [None, None, numpy.int32, numpy.int32, numpy.int32])

    # Create the result image.
    h_output_img = numpy.zeros(img_h * img_w * 1).astype(numpy.uint8)

    # Create the buffers on the device.
    d_input_img = cl.Buffer(context, cl.mem_flags.READ_ONLY |
                            cl.mem_flags.COPY_HOST_PTR, hostbuf=flat_img)

    d_output_img = cl.Buffer(
        context, cl.mem_flags.WRITE_ONLY, h_output_img.nbytes)

    end_setup_grey = time.time()
    benchmark_dataframe["setup_grey"] = [end_setup_grey-start_setup_grey]

    start_grey = time.time()
    #print("Executing kernel grey...")
    kernel_grey(
        queue,
        (img_h, img_w),
        (LOCAL_SIZE_X, LOCAL_SIZE_Y),
        d_input_img,
        d_output_img,
        img_w,
        img_h,
        depth
    )
    end_grey = time.time()
    benchmark_dataframe["grey"] = [end_grey-start_grey]
    #print("Done executing kernel grey.")

    queue.finish()
    cl.enqueue_copy(queue, h_output_img, d_output_img)
    queue.flush()

    start_setup_threshold = time.time()
    LOCAL_SIZE = local_size
    global_size = int((img_w * img_h)/LOCAL_SIZE)

    kernel_threshold.set_scalar_arg_dtypes(
        [None, None,
            numpy.int32, numpy.int32]
    )

    d_input_img = cl.Buffer(
        context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_output_img
    )
    h_input_brightness = numpy.zeros(
        global_size).astype(numpy.uint32)

    d_output_brightness = cl.Buffer(
        context, cl.mem_flags.READ_WRITE, h_input_brightness.nbytes)

    end_setup_threshold = time.time()
    benchmark_dataframe["setup_threshold"] = [
        end_setup_threshold-start_setup_threshold]
    #print("Executing kernel threshold...")
    start_threshold = time.time()
    kernel_threshold(
        queue,
        (global_size, 1),
        (LOCAL_SIZE, 1),
        d_output_img,
        d_output_brightness,
        img_w,
        img_h
    )

    #print("Done executing kernel threshold.")
    queue.finish()
    cl.enqueue_copy(queue, h_input_brightness, d_output_brightness)
    queue.flush()

    #print("Brightness: ", h_input_brightness.sum())
    threshold = 2 * h_input_brightness.sum()/(img_w * img_h)
    threshold = math.floor(threshold)
    end_threshold = time.time()
    benchmark_dataframe["threshold"] = [end_threshold-start_threshold]
    #print("the threshold is: ", threshold)

    # ## Stars
    start_setup_stars = time.time()

    LOCAL_SIZE = local_size
    global_size = int((img_w/LOCAL_SIZE * img_h))

    kernel_stars.set_scalar_arg_dtypes(
        [None, None,
            numpy.int32, numpy.int32, numpy.int32]
    )

    d_input_img = cl.Buffer(
        context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_output_img
    )
    # h_input_count = numpy.zeros(
    #    global_size).astype(numpy.uint32)
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
    #print("Count: ", sum(h_input_count)/255)
    end_total = time.time()
    benchmark_dataframe["total"] = [end_total-start_total]
    benchmark_dataframe["result_threshold"] = [threshold]
    benchmark_dataframe["result_count"] = [count]

    return benchmark_dataframe
