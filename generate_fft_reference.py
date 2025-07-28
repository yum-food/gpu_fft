#!/usr/bin/env python3
"""
Generate reference FFT image in OpenEXR format, matching the shader's output format.
"""

import numpy as np
import argparse
from PIL import Image
import OpenEXR
import Imath

def main():
    parser = argparse.ArgumentParser(description='Generate reference FFT image')
    parser.add_argument('input', type=str, help='Input image file')
    parser.add_argument('output', type=str, help='Output EXR file')
    parser.add_argument('--size', type=int, default=256, help='Size of the FFT (NxN)')
    args = parser.parse_args()

    # Load input image and convert to luminance
    img = Image.open(args.input).convert('RGB')
    img = img.resize((args.size, args.size), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    luminance = 0.2126 * img_array[:,:,0] + 0.7152 * img_array[:,:,1] + 0.0722 * img_array[:,:,2]

    # Perform 2D FFT (no fftshift, matching GPU implementation)
    fft_result = np.fft.fft2(luminance)

    # Pack complex numbers into RGBA matching shader format:
    # R: real part, G: imaginary part, B: 0, A: 1
    real_part = fft_result.real.astype(np.float32)
    imag_part = fft_result.imag.astype(np.float32)

    # Create EXR
    height, width = args.size, args.size
    header = OpenEXR.Header(width, height)
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = {'R': half_chan, 'G': half_chan, 'B': half_chan, 'A': half_chan}

    # Write EXR
    out = OpenEXR.OutputFile(args.output, header)

    # Create zero and one arrays for B and A channels
    zeros = np.zeros((height, width), dtype=np.float32)
    ones = np.ones((height, width), dtype=np.float32)

    out.writePixels({
        'R': real_part.astype(np.float32).tobytes(),
        'G': imag_part.astype(np.float32).tobytes(),
        'B': zeros.tobytes(),
        'A': ones.tobytes()
    })
    out.close()

    print(f"FFT complete. Output saved to {args.output}")
    print(f"Real range: [{real_part.min():.1f}, {real_part.max():.1f}]")
    print(f"Imag range: [{imag_part.min():.1f}, {imag_part.max():.1f}]")

if __name__ == "__main__":
    main()
