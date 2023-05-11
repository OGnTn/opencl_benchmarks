__kernel void kernel_stars(__global unsigned char *input,
                           __global unsigned int *output, const int threshold,
                           const int width, const int height) {

  int gid = get_global_id(0) * get_local_size(0);
  int current_bucket = get_global_id(0);

  int window_size = 3;
  for (int i = 0; i < get_local_size(0); i++) {
    int current_pixel = gid + i;
    // int current_pixel = gid + get_local_id(0);
    int current_pixel_brightness = input[current_pixel];
    int max_brightness_neighbour = 0;

    if (current_pixel_brightness >= threshold) {
      int max_brightness_neighbour = 0;

      for (int x = -window_size; x < window_size + 1; x++) {
        for (int y = -window_size; y < window_size + 1; y++) {

          int next_x = current_pixel + x;
          int row = current_pixel / width;
          int next_row = next_x / width;

          int index = current_pixel + y * width + x;

          if (index < 0 || index >= width * height || row != next_row ||
              (x == 0 && y == 0) ||
              (input[index] <= max_brightness_neighbour)) {
            continue;
          } else {
            max_brightness_neighbour = input[index];
          }
        }
      }
      if (current_pixel_brightness >= max_brightness_neighbour) {
        // output[current_pixel] = 255;
        atomic_add(&output[current_bucket], 255);
      }
    }
  }
}
