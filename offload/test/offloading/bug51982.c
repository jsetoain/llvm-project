// RUN: %libomptarget-compile-generic -O2 && %libomptarget-run-generic
// -O2 to run openmp-opt
// RUN: %libomptarget-compileopt-generic -O2 && %libomptarget-run-generic

int main(void) {
  long int aa = 0;

  int ng = 12;
  int nxyz = 5;

  const long exp = ng * nxyz;

#pragma omp target map(tofrom : aa)
  for (int gid = 0; gid < nxyz; gid++) {
#pragma omp parallel for
    for (unsigned int g = 0; g < ng; g++) {
#pragma omp atomic
      aa += 1;
    }
  }
  if (aa != exp) {
    return 1;
  }
  return 0;
}
