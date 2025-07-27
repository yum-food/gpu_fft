#!/usr/bin/env python3
"""
Generate .cginc file with hard-coded twiddle factor tables for GPU FFT.
Uses full 32-bit float precision instead of textures.
Uses static branching with preprocessor directives.
"""

import math
import os

def twiddle(k, N):
    """Compute twiddle factor W_N^k = exp(-2*pi*i*k/N)"""
    angle = -2.0 * math.pi * k / N
    return complex(math.cos(angle), math.sin(angle))

def generate_dft_matrix(radix):
    """Generate DFT matrix for given radix"""
    matrix = []
    for k in range(radix):
        row = []
        for n in range(radix):
            row.append(twiddle(k * n, radix))
        matrix.append(row)
    return matrix

def format_complex(c):
    """Format complex number as float2"""
    return f"float2({c.real:.9f}f, {c.imag:.9f}f)"

def generate_cginc(radices, max_size, output_file):
    """Generate .cginc file with twiddle factor tables"""

    with open(output_file, 'w') as f:
        f.write("// Auto-generated FFT twiddle factor tables\n")
        f.write("#ifndef FFT_TWIDDLE_TABLES_CGINC\n")
        f.write("#define FFT_TWIDDLE_TABLES_CGINC\n\n")

        # Generate DFT matrices for each radix
        for radix in radices:
            f.write(f"#if defined(GPU_FFT_RADIX{radix})\n")
            f.write(f"static const float2 DFT_MATRIX[{radix}][{radix}] = {{\n")

            matrix = generate_dft_matrix(radix)
            for k in range(radix):
                f.write("    { ")
                for n in range(radix):
                    f.write(format_complex(matrix[k][n]))
                    if n < radix - 1:
                        f.write(", ")
                f.write(" }")
                if k < radix - 1:
                    f.write(",")
                f.write("\n")
            f.write("};\n")
            f.write("#endif\n\n")

        # Generate stage twiddle factor tables
        f.write("// Stage twiddle factors\n")
        for radix in radices:
            butterfly_size = radix
            stage_idx = 0

            while butterfly_size <= max_size:
                f.write(f"#if defined(GPU_FFT_RADIX{radix}_SIZE{butterfly_size})\n")
                f.write(f"static const float2 STAGE_TWIDDLES[{butterfly_size}] = {{\n")

                # Write twiddles in rows of 4 for readability
                for i in range(0, butterfly_size, 4):
                    f.write("    ")
                    for j in range(4):
                        if i + j < butterfly_size:
                            c = twiddle(i + j, butterfly_size)
                            f.write(format_complex(c))
                            if i + j < butterfly_size - 1:
                                f.write(", ")
                    f.write("\n")

                f.write("};\n")
                f.write("#endif\n\n")
                butterfly_size *= radix
                stage_idx += 1

        f.write("#endif // FFT_TWIDDLE_TABLES_CGINC\n\n")

def main():
    radices = [2, 4, 8, 16]
    max_size = 1024
    output_file = 'fft_twiddle_tables.cginc'

    print(f"Generating twiddle factor tables for radices: {radices}")
    print(f"Maximum FFT size: {max_size}")
    print(f"Output file: {output_file}")

    generate_cginc(radices, max_size, output_file)

if __name__ == "__main__":
    main()

