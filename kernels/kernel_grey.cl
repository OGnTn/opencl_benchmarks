__kernel void kernel_grey(__global unsigned char *input,
                          __global unsigned char *output, const int width,
                          const int height, const int depth) {
                            
  int i = get_global_id(0);
  int j = get_global_id(1);
  float l = 0;

  l = 0.299 * input[(i * width * depth + j * depth)] +
      0.587 * input[(i * width * depth + j * depth) + 1] +
      0.114 * input[(i * width * depth + j * depth) + 2];
  output[(i * width + j)] = round(l);

  
}
