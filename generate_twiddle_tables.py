#!/usr/bin/env python3
"""
Generate .cginc file with hard-coded twiddle factor tables for GPU FFT.
Uses full 32-bit float precision instead of textures.
Uses static branching with preprocessor directives.
"""

import math
import os

def twiddle(k, N, inverse=False):
    """Compute twiddle factor W_N^k = exp(-2*pi*i*k/N) for forward FFT
    or exp(+2*pi*i*k/N) for inverse FFT"""
    angle = (2.0 if inverse else -2.0) * math.pi * k / N
    return complex(math.cos(angle), math.sin(angle))

def generate_dft_matrix(radix, inverse=False):
    """Generate DFT matrix for given radix"""
    matrix = []
    for k in range(radix):
        row = []
        for n in range(radix):
            row.append(twiddle(k * n, radix, inverse))
        matrix.append(row)
    return matrix

def format_complex(c):
    """Format complex number as float2"""
    return f"float2({c.real}f, {c.imag}f)"

def get_butterfly_sizes_for_config(n, radix):
    """Get the exact butterfly sizes needed for a specific N and radix"""
    butterfly_sizes = []
    num_stages = int(math.log(n) / math.log(radix))

    for stage in range(num_stages):
        span = n // (radix ** (stage + 1))
        butterfly_size = span * radix
        butterfly_sizes.append(butterfly_size)

    return butterfly_sizes

def generate_cginc(radices, max_size, output_file):
    """Generate .cginc file with twiddle factor tables"""

    with open(output_file, 'w') as f:
        f.write("// Auto-generated FFT twiddle factor tables\n")
        f.write("#ifndef FFT_TWIDDLE_TABLES_CGINC\n")
        f.write("#define FFT_TWIDDLE_TABLES_CGINC\n\n")

        # Generate DFT matrices for each radix (forward)
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

            # Generate inverse DFT matrix
            f.write(f"static const float2 IDFT_MATRIX[{radix}][{radix}] = {{\n")
            matrix_inv = generate_dft_matrix(radix, inverse=True)
            for k in range(radix):
                f.write("    { ")
                for n in range(radix):
                    f.write(format_complex(matrix_inv[k][n]))
                    if n < radix - 1:
                        f.write(", ")
                f.write(" }")
                if k < radix - 1:
                    f.write(",")
                f.write("\n")
            f.write("};\n")
            f.write("#endif\n\n")

        # Generate stage twiddle tables for each radix/size combination
        f.write("// Stage twiddle factors for specific radix/size combinations\n")

        for radix in radices:
            n = radix
            while n <= max_size:
                # Get the exact butterfly sizes for this configuration
                butterfly_sizes = get_butterfly_sizes_for_config(n, radix)

                # Generate tables for this specific configuration
                f.write(f"\n#if defined(GPU_FFT_RADIX{radix}_N{n})\n")

                # Generate a table for each stage
                for stage_idx, butterfly_size in enumerate(butterfly_sizes):
                    f.write(f"// Stage {stage_idx}: butterfly_size = {butterfly_size}\n")
                    f.write(f"static const float2 STAGE{stage_idx}_TWIDDLES[{butterfly_size}] = {{\n")

                    for i in range(0, butterfly_size, 4):
                        f.write("    ")
                        line_items = []
                        for j in range(4):
                            if i + j < butterfly_size:
                                c = twiddle(i + j, butterfly_size)
                                line_items.append(format_complex(c))
                        f.write(", ".join(line_items))
                        if i + 4 < butterfly_size:
                            f.write(",")
                        f.write("\n")
                    f.write("};\n")

                    # Inverse version
                    f.write(f"static const float2 STAGE{stage_idx}_TWIDDLES_INV[{butterfly_size}] = {{\n")
                    for i in range(0, butterfly_size, 4):
                        f.write("    ")
                        line_items = []
                        for j in range(4):
                            if i + j < butterfly_size:
                                c = twiddle(i + j, butterfly_size, inverse=True)
                                line_items.append(format_complex(c))
                        f.write(", ".join(line_items))
                        if i + 4 < butterfly_size:
                            f.write(",")
                        f.write("\n")
                    f.write("};\n\n")

                f.write("#endif\n")
                n *= radix


        f.write("#endif // FFT_TWIDDLE_TABLES_CGINC\n\n")

def main():
    radices = [2, 4, 8, 16]
    max_size = 1024
    output_file = 'fft_twiddle_tables.cginc'

    print(f"Generating twiddle factor tables for radices: {radices}")
    print(f"Maximum FFT size: {max_size}")
    print(f"Output file: {output_file}")

    generate_cginc(radices, max_size, output_file)

    # Print summary of what was generated
    print("\nGenerated configurations:")
    for radix in radices:
        print(f"\n  Radix {radix}:")
        n = radix
        while n <= max_size:
            butterfly_sizes = get_butterfly_sizes_for_config(n, radix)
            print(f"    N={n}: stages use butterfly sizes {butterfly_sizes}")
            n *= radix

if __name__ == "__main__":
    main()

