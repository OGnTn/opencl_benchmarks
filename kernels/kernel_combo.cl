__kernel void kernel_combo(__global unsigned char *input,
                           __global unsigned char *output,
                           __global unsigned int *brightness, const int width,
                           const int height, const int depth) {

  float l = 0;

  int y = get_global_id(0);
  int x = get_global_id(1);
  l = 0.299 * input[(y * width * depth + x * depth)] +
      0.587 * input[(y * width * depth + x * depth) + 1] +
      0.114 * input[(y * width * depth + x * depth) + 2];
  output[(y * width + x)] = round(l);
  // atomic_add(&output[y * width + x], round(l));
  atomic_add(&brightness[y], round(l));
  // brightness[y] += round(l);
}
