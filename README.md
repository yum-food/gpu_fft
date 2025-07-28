## FFT on the GPU

This is an optimized GPU-based 2D FFT for VRChat. Since it uses RenderTextures,
it is only suitable for use in worlds.

Please take note that the FFT is generated in bit-reversed order. You'll have
to bit-reverse your indices when accessing it. See `apply_2d_bit_reversal` in
`gpu_fft.cc` for an example of how to do that.

### Quick start

Clone the repo into your unity project. Open the scene.

Run CPU simulator:

```bash
$ cmake .. && cmake --build . && ./gpu_fft
```

Generate twiddle factor tables:

```bash
$ python3 ./generate_twiddle_tables.py
```

### Overview

`gpu_fft.cc` is a CPU simulator achieving high performance. It compares the GPU
algorithm against a simple radix-2 algorithm, demonstrating agreement within
some modest epsilons. Because higher radix FFTs do more sequential adds than
lower ffts, there is substantial error. In exchange, higher radices let you
compute FFTs with a shorter CRT chain.

`generate_twiddle_tables.py` generates precomputes twiddle factors.
`DFT_MATRIX` corresponds to `ShaderUniforms.twiddle_factors` in the simulator
and `STAGE_TWIDDLES` corresponds to `ShaderUniforms.stage_twiddles`.

