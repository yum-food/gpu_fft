// The goal of this code is to demonstrate how we're going to implement the FFT
// on the GPU. The core idea is to divide the work into log(N) stages, each stage
// doing O(N) work. This work is all done in the pixel shader, so each stage is
// computed in parallel. This should permit us to crunch large FFTs in real time.
// An optimization is to use radix-16, so we can crunch a 256x256 FFT in 4 stages.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <random>
#include <vector>

struct float2 {
  float x, y;
};
#define GPU_FFT_RADIX16
#define GPU_FFT_RADIX16_N256
#include "fft_twiddle_tables.cginc"

// This is a reference Cooley-Tukey FFT implementation. It's just here to check
// for correctness.
std::vector<std::complex<float>> fft1d_naive(
    const std::vector<std::complex<float>>& data) {
  const int N = (int) data.size();
  if (N == 1) {
    return data;
  }

  std::vector<std::complex<float>> even(N/2), odd(N/2);
  for (int i = 0; i < N; ++i) {
    if (i % 2 == 0) {
      even[i/2] = data[i];
    } else {
      odd[i/2] = data[i];
    }
  }

  std::vector<std::complex<float>> Fe = fft1d_naive(even);
  std::vector<std::complex<float>> Fo = fft1d_naive(odd);

  std::vector<std::complex<float>> F(N);
  for (int k = 0; k < N/2; ++k) {
    float angle = -2.0f * std::numbers::pi * k / N;
    std::complex<float> w(std::cos(angle), std::sin(angle));
    F[k]       = Fe[k] + w * Fo[k];
    F[k+N/2] = Fe[k] - w * Fo[k];
  }
  return F;
}

std::vector<std::vector<std::complex<float>>> fft2d_naive(
    const std::vector<std::vector<std::complex<float>>> &data) {
  const int rows = data.size();
  const int cols = data[0].size();

  // FFT of rows
  std::vector<std::vector<std::complex<float>>> temp(rows, std::vector<std::complex<float>>(cols));
  for (int i = 0; i < rows; ++i)
  {
    temp[i] = fft1d_naive(data[i]);
  }

  // FFT of columns
  std::vector<std::vector<std::complex<float>>> result(rows, std::vector<std::complex<float>>(cols));
  for (int j = 0; j < cols; ++j)
  {
    std::vector<std::complex<float>> col(rows);
    for (int i = 0; i < rows; ++i)
      col[i] = temp[i][j];
    col = fft1d_naive(col);
    for (int i = 0; i < rows; ++i)
      result[i][j] = col[i];
  }

  return result;
}

// GPU FFT implementation types and structures
typedef std::pair<float, float> gpu_complex;
typedef std::vector<std::vector<gpu_complex>> stage_texture;
typedef std::pair<int, int> pixel_index;

// Shader uniforms - data that's constant for all pixels in a stage
struct ShaderUniforms {
  // These will be passed as material properties.
  int n;
  int radix;
  int stage;
  int num_stages_per_dim;
  int span;
  int butterfly_size;
  bool inverse;
  // This will be baked into a texture.
  std::vector<std::vector<gpu_complex>> twiddle_factors;
  // Precomputed stage twiddle factors
  std::vector<gpu_complex> stage_twiddles;
};

// Generalized digit reversal for any radix.
unsigned int reverse_digits(unsigned int n, unsigned int num_digits, unsigned int radix) {
  const unsigned int bits_per_digit = std::log2(radix);
  const unsigned int digit_mask = radix - 1;
  unsigned int reversed = 0;

  for (unsigned int i = 0; i < num_digits; ++i) {
    unsigned int digit = (n >> (bits_per_digit * i)) & digit_mask;
    reversed |= digit << (bits_per_digit * (num_digits - 1 - i));
  }
  return reversed;
}

// Compute twiddle factor W_N^k = exp(-2*pi*i*k/N) for forward FFT
// or exp(+2*pi*i*k/N) for inverse FFT
gpu_complex twiddle_factor(int k, int N, bool inverse = false) {
  // Use double precision for angle computation, then cast to float
  // This matches how the precomputed tables were generated
  double angle = (inverse ? 2.0 : -2.0) * std::numbers::pi * k / N;
  return {(float)std::cos(angle), (float)std::sin(angle)};
}

// Helper to compute radix^power using integer arithmetic
int int_pow(int base, int exp) {
  int result = 1;
  for (int i = 0; i < exp; ++i) {
    result *= base;
  }
  return result;
}

// Apply 2D bit reversal to the output
void apply_2d_bit_reversal(const int n, const int radix, const stage_texture& in, stage_texture& out) {
  const int num_digits = std::log2(n) / std::log2(radix);

  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      const int rev_x = reverse_digits(x, num_digits, radix);
      const int rev_y = reverse_digits(y, num_digits, radix);
      out[y][x] = in[rev_y][rev_x];
    }
  }
}

// Main shader function - simplified for GPU/HLSL conversion
void shader_stage(
    const ShaderUniforms& uniforms,
    const pixel_index& px,
    const stage_texture& in,
    gpu_complex& out) {
  // Extract coordinates
  const int x = px.first;
  const int y = px.second;

  // Determine processing direction
  const bool is_row_stage = (uniforms.stage < uniforms.num_stages_per_dim);
  const int coord = is_row_stage ? x : y;

  // Calculate butterfly indices (simple integer math)
  const int group = coord / uniforms.butterfly_size;
  const int idx_in_group = coord % uniforms.butterfly_size;
  const int wing = idx_in_group / uniforms.span;
  const int idx_in_wing = idx_in_group % uniforms.span;

  // Accumulate DFT sum
  float sum_real = 0.0f;
  float sum_imag = 0.0f;

  // Main DFT loop
  for (int i = 0; i < uniforms.radix; ++i) {
    // Calculate input position
    int input_pos = group * uniforms.butterfly_size + i * uniforms.span + idx_in_wing;

    // Read input value
    float in_real = is_row_stage ? in[y][input_pos].first : in[input_pos][x].first;
    float in_imag = is_row_stage ? in[y][input_pos].second : in[input_pos][x].second;

    // Read DFT coefficient
    float coeff_real = uniforms.twiddle_factors[wing][i].first;
    float coeff_imag = uniforms.twiddle_factors[wing][i].second;

    // Complex multiply-accumulate
    sum_real += coeff_real * in_real - coeff_imag * in_imag;
    sum_imag += coeff_real * in_imag + coeff_imag * in_real;
  }

  // Apply stage twiddle if needed
  if (wing > 0 && idx_in_wing > 0) {
    int twiddle_idx = wing * idx_in_wing;
    float tw_real = uniforms.stage_twiddles[twiddle_idx].first;
    float tw_imag = uniforms.stage_twiddles[twiddle_idx].second;

    // Output = twiddle * sum
    out.first = tw_real * sum_real - tw_imag * sum_imag;
    out.second = tw_real * sum_imag + tw_imag * sum_real;
  } else {
    out.first = sum_real;
    out.second = sum_imag;
  }
}

// Evalaute one stage.
void evaluate_stage(
    const ShaderUniforms& uniforms,
    const stage_texture& in,
    stage_texture& out) {
  for (int y = 0; y < uniforms.n; ++y) {
    for (int x = 0; x < uniforms.n; ++x) {
      shader_stage(uniforms, {x, y}, in, out[y][x]);
    }
  }
}

// Evaluate all stages - unified function
void evaluate_stages(
    const int n,
    const int radix,
    const bool inverse,
    std::vector<stage_texture>& textures,
    const std::vector<std::vector<gpu_complex>>& dft_matrix,
    const std::vector<std::vector<gpu_complex>>& stage_twiddles_array) {

  const int num_stages_per_dim = std::log2(n) / std::log2(radix);
  const int num_stages = num_stages_per_dim * 2;

  for (int stage = 0; stage < num_stages; ++stage) {
    int current_stage = (stage < num_stages_per_dim) ? stage : (stage - num_stages_per_dim);
    int span = n / int_pow(radix, current_stage + 1);
    int butterfly_size = span * radix;

    const std::vector<gpu_complex>& stage_twiddles = stage_twiddles_array[current_stage];

    ShaderUniforms uniforms = {n, radix, stage, num_stages_per_dim, span, butterfly_size,
                               inverse, dft_matrix, stage_twiddles};
    evaluate_stage(uniforms, textures[stage], textures[stage+1]);
  }

  // Apply bit reversal once at the end
  stage_texture temp = textures[num_stages];
  apply_2d_bit_reversal(n, radix, temp, textures[num_stages]);

  // For inverse FFT, normalize by 1/(n*n)
  if (inverse) {
    float norm_factor = 1.0f / (n * n);
    for (int y = 0; y < n; ++y) {
      for (int x = 0; x < n; ++x) {
        textures[num_stages][y][x].first *= norm_factor;
        textures[num_stages][y][x].second *= norm_factor;
      }
    }
  }
}

// Precompute twiddle factors for a given radix
std::vector<std::vector<gpu_complex>> compute_twiddle_factors(int radix, bool inverse = false) {
  std::vector<std::vector<gpu_complex>> twiddle_factors(radix,
      std::vector<gpu_complex>(radix));
  for (int k = 0; k < radix; ++k) {
    for (int n = 0; n < radix; ++n) {
      twiddle_factors[k][n] = twiddle_factor(k * n, radix, inverse);
    }
  }
  return twiddle_factors;
}

// Precompute stage twiddle factors
std::vector<gpu_complex> compute_stage_twiddles(int butterfly_size, bool inverse = false) {
  std::vector<gpu_complex> stage_twiddles(butterfly_size);
  for (int i = 0; i < butterfly_size; ++i) {
    stage_twiddles[i] = twiddle_factor(i, butterfly_size, inverse);
  }
  return stage_twiddles;
}

// Convert float2 arrays to gpu_complex vectors
std::vector<std::vector<gpu_complex>> convert_dft_matrix(bool inverse = false) {
  std::vector<std::vector<gpu_complex>> result(16, std::vector<gpu_complex>(16));
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      if (inverse) {
        result[i][j] = {IDFT_MATRIX[i][j].x, IDFT_MATRIX[i][j].y};
      } else {
        result[i][j] = {DFT_MATRIX[i][j].x, DFT_MATRIX[i][j].y};
      }
    }
  }
  return result;
}

std::vector<gpu_complex> convert_stage_twiddles(int stage, bool inverse = false) {
  if (stage == 0) { // butterfly_size = 256
    std::vector<gpu_complex> result(256);
    for (int i = 0; i < 256; ++i) {
      if (inverse) {
        result[i] = {STAGE0_TWIDDLES_INV[i].x, STAGE0_TWIDDLES_INV[i].y};
      } else {
        result[i] = {STAGE0_TWIDDLES[i].x, STAGE0_TWIDDLES[i].y};
      }
    }
    return result;
  } else { // stage == 1, butterfly_size = 16
    std::vector<gpu_complex> result(16);
    for (int i = 0; i < 16; ++i) {
      if (inverse) {
        result[i] = {STAGE1_TWIDDLES_INV[i].x, STAGE1_TWIDDLES_INV[i].y};
      } else {
        result[i] = {STAGE1_TWIDDLES[i].x, STAGE1_TWIDDLES[i].y};
      }
    }
    return result;
  }
}

// Wrapper for computed twiddles
void evaluate_stages_computed(
    const int n,
    const int radix,
    const bool inverse,
    std::vector<stage_texture>& textures) {

  const std::vector<std::vector<gpu_complex>> dft_matrix =
    compute_twiddle_factors(radix, inverse);

  // Precompute all stage twiddles
  const int num_stages_per_dim = std::log2(n) / std::log2(radix);
  std::vector<std::vector<gpu_complex>> stage_twiddles_array(num_stages_per_dim);

  for (int stage = 0; stage < num_stages_per_dim; ++stage) {
    int span = n / int_pow(radix, stage + 1);
    int butterfly_size = span * radix;
    stage_twiddles_array[stage] = compute_stage_twiddles(butterfly_size, inverse);
  }

  evaluate_stages(n, radix, inverse, textures, dft_matrix, stage_twiddles_array);
}

// Wrapper for precomputed twiddles
void evaluate_stages_precomputed(
    const int n,
    const int radix,
    const bool inverse,
    std::vector<stage_texture>& textures,
    const std::vector<std::vector<gpu_complex>>& precomputed_dft_matrix) {

  // Convert precomputed stage twiddles
  const int num_stages_per_dim = std::log2(n) / std::log2(radix);
  std::vector<std::vector<gpu_complex>> stage_twiddles_array(num_stages_per_dim);

  for (int stage = 0; stage < num_stages_per_dim; ++stage) {
    stage_twiddles_array[stage] = convert_stage_twiddles(stage, inverse);
  }

  evaluate_stages(n, radix, inverse, textures, precomputed_dft_matrix, stage_twiddles_array);
}

// Verify FFT results match between computed and precomputed tables
bool verify_fft_with_tables(std::mt19937& rng) {
  const int n = 256;
  const int radix = 16;
  const int NUM_STAGES = (std::log2(n) / std::log2(radix)) * 2;

  // Initialize test data
  const std::vector<std::vector<gpu_complex>> black_texture(n,
      std::vector<gpu_complex>(n, {0, 0}));
  std::vector<stage_texture> textures_computed(NUM_STAGES + 1, black_texture);
  std::vector<stage_texture> textures_precomputed(NUM_STAGES + 1, black_texture);

  // Fill with identical random data
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      float real = dist(rng);
      float imag = dist(rng);
      textures_computed[0][y][x] = {real, imag};
      textures_precomputed[0][y][x] = {real, imag};
    }
  }

  // Run FFT with computed tables
  evaluate_stages_computed(n, radix, false, textures_computed);

  // Run FFT with precomputed tables from cginc
  auto dft_matrix = convert_dft_matrix(false);
  evaluate_stages_precomputed(n, radix, false, textures_precomputed, dft_matrix);

  // Compare results
  float max_error = 0.0f;
  int mismatch_count = 0;
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      float err_real = std::abs(textures_computed[NUM_STAGES][y][x].first -
                                textures_precomputed[NUM_STAGES][y][x].first);
      float err_imag = std::abs(textures_computed[NUM_STAGES][y][x].second -
                                textures_precomputed[NUM_STAGES][y][x].second);
      max_error = std::max(max_error, std::max(err_real, err_imag));
      if (err_real > 1e-6f || err_imag > 1e-6f) {
        mismatch_count++;
      }
    }
  }

  std::cout << "FFT max error between computed and precomputed tables: "
            << std::scientific << max_error << std::fixed << std::endl;

  if (mismatch_count > 0) {
    std::cout << "ERROR: " << mismatch_count << " pixels have error > 1e-6" << std::endl;
    return false;
  }

  return true;
}

bool check_result(
    const int n,
    const stage_texture& gpu_result,
    std::vector<std::vector<std::complex<float>>>& reference_result,
    const float epsilon,
    bool print_correctness_matrix) {
  bool ret = true;
  std::vector<std::vector<bool>> is_ok(n, std::vector<bool>(n, true));

  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      if (std::abs(gpu_result[y][x].first - reference_result[y][x].real()) > epsilon) {
        ret = false;
        is_ok[y][x] = false;
      }
      if (std::abs(gpu_result[y][x].second - reference_result[y][x].imag()) > epsilon) {
        ret = false;
        is_ok[y][x] = false;
      }
    }
  }

  if (print_correctness_matrix) {
    for (int y = 0; y < n; ++y) {
      for (int x = 0; x < n; ++x) {
        std::cout << is_ok[y][x] << "";
      }
      std::cout << std::endl;
    }
  }

  return ret;
}

float compute_max_error(
    const int n,
    const stage_texture& gpu_result,
    const std::vector<std::vector<std::complex<float>>>& reference_result) {
  float max_error = 0.0f;
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      float err_real = std::abs(gpu_result[y][x].first - reference_result[y][x].real());
      float err_imag = std::abs(gpu_result[y][x].second - reference_result[y][x].imag());
      max_error = std::max(max_error, std::max(err_real, err_imag));
    }
  }
  return max_error;
}

void print_diagnostics(
    const int n,
    const stage_texture& input,
    const stage_texture& gpu_result,
    const std::vector<std::vector<std::complex<float>>>& reference_result) {
  auto print_blocks = [&](const std::string& title, auto get_val) {
    std::cout << "\n" << title << ":" << std::endl;
    for (int y = 0; y < std::min(4, n); ++y) {
      for (int x = 0; x < std::min(4, n); ++x) {
        auto val = get_val(y, x);
        std::cout << std::fixed << std::setprecision(1) << std::setw(8) << val << " ";
      }
      std::cout << std::endl;
    }
  };

  print_blocks("Input", [&](int y, int x) { return input[y][x].first; });

  print_blocks("Reference", [&](int y, int x) { return reference_result[y][x].real(); });

  print_blocks("GPU", [&](int y, int x) { return gpu_result[y][x].first; });

  print_blocks("Delta", [&](int y, int x) {
      return gpu_result[y][x].first - reference_result[y][x].real();
      });
}

bool evaluateAlgorithm(const int n, const int radix, const bool inverse, std::mt19937& rng) {
  const int NUM_STAGES = (std::log2(n) / std::log2(radix)) * 2;

  const std::vector<std::vector<gpu_complex>> black_texture(n,
      std::vector<gpu_complex>(n, {0, 0}));
  std::vector<stage_texture> textures(NUM_STAGES + 1, black_texture);

  // Initialize the input texture with random data.
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      textures[0][y][x] = {dist(rng), dist(rng)};
    }
  }

  // Evaluate the GPU algorithm.
  auto start = std::chrono::high_resolution_clock::now();
  evaluate_stages_computed(n, radix, inverse, textures);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Do the same thing with the reference algorithm.
  std::vector<std::vector<std::complex<float>>> reference_input(n,
      std::vector<std::complex<float>>(n, {0, 0}));
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      reference_input[y][x].real(textures[0][y][x].first);
      reference_input[y][x].imag(textures[0][y][x].second);
    }
  }
  auto ref_start = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<std::complex<float>>> reference_result = fft2d_naive(reference_input);
  auto ref_end = std::chrono::high_resolution_clock::now();
  auto ref_dur = std::chrono::duration_cast<std::chrono::microseconds>(ref_end - ref_start);

  std::cout << "runtime: " << duration.count() << " μs (ref: " << ref_dur.count()
            << " μs, ratio: " << std::fixed << std::setprecision(2)
            << (float)duration.count() / ref_dur.count() << "x)" << std::endl;

  // Check the result. Note the increased epsilon due to float precision differences.
  // Higher radix values accumulate more floating point error
  const float epsilon = 1e-1;
  if (check_result(n, textures[NUM_STAGES], reference_result, epsilon, false)) {
    return true;
  }

  std::cout << "The result is incorrect." << std::endl;
  print_diagnostics(n, textures[0], textures[NUM_STAGES], reference_result);
  float max_error = compute_max_error(n, textures[NUM_STAGES], reference_result);
  std::cout << "Max error: " << std::setprecision(5) << max_error << std::endl;
  return false;
}

// Test FFT followed by inverse FFT
bool testFFTInverse(const int n, const int radix, std::mt19937& rng) {
  const int NUM_STAGES = (std::log2(n) / std::log2(radix)) * 2;

  const std::vector<std::vector<gpu_complex>> black_texture(n,
      std::vector<gpu_complex>(n, {0, 0}));
  std::vector<stage_texture> textures_fft(NUM_STAGES + 1, black_texture);
  std::vector<stage_texture> textures_ifft(NUM_STAGES + 1, black_texture);

  // Initialize with random data
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      textures_fft[0][y][x] = {dist(rng), dist(rng)};
    }
  }

  // Save original input
  stage_texture original_input = textures_fft[0];

  // Perform forward FFT
  evaluate_stages_computed(n, radix, false, textures_fft);

  // Copy FFT result as input to inverse FFT
  textures_ifft[0] = textures_fft[NUM_STAGES];

  // Perform inverse FFT
  evaluate_stages_computed(n, radix, true, textures_ifft);

  // Check if inverse FFT gives back the original input
  float max_error = 0.0f;
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      float err_real = std::abs(textures_ifft[NUM_STAGES][y][x].first - original_input[y][x].first);
      float err_imag = std::abs(textures_ifft[NUM_STAGES][y][x].second - original_input[y][x].second);
      max_error = std::max(max_error, std::max(err_real, err_imag));
    }
  }

  const float epsilon = 1e-5; // Tolerance for round-trip error
  bool success = (max_error < epsilon);

  if (!success) {
    std::cout << "FFT->IFFT round-trip test FAILED. Max error: " << max_error << std::endl;

    // Print some diagnostics
    std::cout << "\nFirst 4x4 block comparison:" << std::endl;
    std::cout << "Original vs Reconstructed (real parts):" << std::endl;
    for (int y = 0; y < std::min(4, n); ++y) {
      for (int x = 0; x < std::min(4, n); ++x) {
        std::cout << std::fixed << std::setprecision(3)
                  << original_input[y][x].first << " ";
      }
      std::cout << " | ";
      for (int x = 0; x < std::min(4, n); ++x) {
        std::cout << std::fixed << std::setprecision(3)
                  << textures_ifft[NUM_STAGES][y][x].first << " ";
      }
      std::cout << std::endl;
    }
  }

  return success;
}

int main() {
  std::mt19937 rng(std::random_device{}());

  // Verify FFT results match with precomputed tables
  std::cout << "Verifying FFT with precomputed tables from fft_twiddle_tables.cginc..." << std::endl;
  if (!verify_fft_with_tables(rng)) {
    std::cout << "ERROR: FFT results do not match between computed and precomputed tables!" << std::endl;
    return 1;
  }
  std::cout << "FFT verification passed!\n" << std::endl;

  // First run the original forward FFT tests
  std::cout << "Testing forward FFT correctness against reference implementation..." << std::endl;
  for (int log_radix = 1; log_radix < 5; ++log_radix) {
    int radix = std::pow(2, log_radix);
    for (int log_n = 1; log_n < 12; ++log_n) {
      int n = std::pow(radix, log_n);
      if (n > 1024) {
        break;
      }
      std::cout << "Testing radix=" << radix << " n=" << n << std::endl;
      if (!evaluateAlgorithm(n, radix, false, rng)) {
        return 1;
      }
    }
  }

  std::cout << "\nAll forward FFT tests passed!" << std::endl;

  // Now run the FFT->IFFT round-trip tests
  std::cout << "\nTesting FFT->IFFT round-trip correctness..." << std::endl;

  for (int log_radix = 1; log_radix < 5; ++log_radix) {
    int radix = std::pow(2, log_radix);
    for (int log_n = 1; log_n < 10; ++log_n) {
      int n = std::pow(radix, log_n);
      if (n > 512) {
        break;
      }
      std::cout << "Testing radix=" << radix << " n=" << n << " ... ";
      if (testFFTInverse(n, radix, rng)) {
        std::cout << "PASSED" << std::endl;
      } else {
        std::cout << "FAILED" << std::endl;
        return 1;
      }
    }
  }

  std::cout << "\nAll FFT->IFFT tests passed!" << std::endl;
  return 0;
}
