#### IMPORTS ####
import os
import pandas as pd
import implementations.naive as naive
import implementations.optimized as optimized

#### CONSTANTS ####
results_path = "results"
images_path = "images"
ITER_AMOUNT = 30

### BENCHMARK_CASES ###


def ls_naive_vs_optimized():
    # naive vs optimized for varying local size
    ls = [1, 2, 4, 5, 16, 32]
    path = images_path + "/" + "andromeda-2.jpg"
    naive.set_context("0", path)
    optimized.set_context("0", path)
    for i in ls:
        benchmark("case_ls/naive_andromeda2_" + str(i), naive.run_naive, i)
        benchmark("case_ls/optimized_andromeda2_" + str(i), optimized.run_optimized, i)


def all_images():
    ls = 5
    for image in os.listdir(images_path):
        naive.set_context("0", images_path + "/" + image)
        optimized.set_context("0", images_path + "/" + image)
        benchmark("case_all/naive_" + os.path.splitext(image)[0], naive.run_naive, ls)
        benchmark(
            "case_all/optimized_" + os.path.splitext(image)[0],
            optimized.run_optimized,
            ls,
        )


### RUN_BENCHMARKS ###


def benchmark(name, function, *args):
    times = pd.DataFrame()
    for i in range(ITER_AMOUNT):
        times = pd.concat([times, function(*args)], ignore_index=True)
    times.to_csv(results_path + "/" + name + ".csv")


def run_bencharks():
    ls_naive_vs_optimized()
    all_images()


run_bencharks()

# naive vs optimized, varying work group size, gpu

# gpu vs cpu

# biggest image vs smallest image
