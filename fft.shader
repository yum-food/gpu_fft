Shader "yum_food/fft"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _N ("N", Int) = 256
        _Radix ("Radix", Int) = 16
        _Stage ("Stage", Int) = 0
        [Toggle] _Passthrough ("Pass Through", Float) = 0
        [Toggle] _LDS ("Temporal LDS", Float) = 0
        [Toggle] _Luminance ("Luminance", Float) = 0
        [Toggle] _Inverse ("Inverse FFT", Float) = 0
        [Toggle] _BitReversal ("Bit Reversal Only", Float) = 0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"
            #define GPU_FFT_RADIX16
            #define GPU_FFT_RADIX16_N256
            #include "fft_twiddle_tables.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
                uint num_stages_per_dim : TEXCOORD1;
                int span : TEXCOORD2;
                int butterfly_size : TEXCOORD3;
                int num_stages : TEXCOORD4;
            };

            texture2D _MainTex;
            SamplerState point_clamp_s;
            int _N;
            int _Radix;
            int _Stage;
            float _Passthrough;
            float _LDS;
            float _Luminance;
            float _Inverse;
            float _BitReversal;

            #define PHI 1.618033988749894

            // Helper function to compute integer power
            int int_pow(int base, int exp)
            {
                int result = 1;
                for (int i = 0; i < exp; i++)
                {
                    result *= base;
                }
                return result;
            }

            // Generalized digit reversal for any radix
            uint reverse_digits(uint n, uint num_digits, uint radix)
            {
                uint bits_per_digit = (uint)(log2(radix));
                uint digit_mask = radix - 1;
                uint reversed = 0;

                for (uint i = 0; i < num_digits; i++)
                {
                    uint digit = (n >> (bits_per_digit * i)) & digit_mask;
                    reversed |= digit << (bits_per_digit * (num_digits - 1 - i));
                }
                return reversed;
            }

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;

                // Calculate num_stages_per_dim = log_radix(N)
                o.num_stages_per_dim = (uint)(log(_N) / log(_Radix));

                // Determine current stage (0-based index within row or column passes)
                int current_stage = (_Stage < o.num_stages_per_dim) ? _Stage : (_Stage - o.num_stages_per_dim);

                // Calculate span and butterfly_size
                o.span = _N / int_pow(_Radix, current_stage + 1);
                o.butterfly_size = o.span * _Radix;

                return o;
            }

            float luminance(float3 color) {
              return dot(color, float3(0.2126, 0.7152, 0.0722));
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // Extract coordinates
                int2 pixel_index = (int2)(i.uv * _N);
                int x = pixel_index.x;
                int y = pixel_index.y;

                // Bit reversal mode
                if (_BitReversal > 0.5)
                {
                    uint num_digits = i.num_stages_per_dim;
                    uint rev_x = reverse_digits((uint)x, num_digits, (uint)_Radix);
                    uint rev_y = reverse_digits((uint)y, num_digits, (uint)_Radix);

                    float2 rev_uv = float2((rev_x + 0.5) / (float) _N, (rev_y + 0.5) / (float) _N);
                    float4 col = _MainTex.SampleLevel(point_clamp_s, rev_uv, 0);
                    return col;
                }

                // Pass through mode
                if (_Passthrough > 0.5)
                {
                  float2 uv = (pixel_index + 0.5f) / _N;
                  float3 col = _MainTex.SampleLevel(point_clamp_s, uv, 0).rgb;
                  if (_LDS > 0.5) {
                    col += PHI * _Time[0];
                    col = frac(col);
                  }
                  if (_Luminance > 0.5) {
                    col = luminance(col);
                  }
                  return float4(col, 1);
                }

                // Determine processing direction
                bool is_row_stage = (_Stage < i.num_stages_per_dim);
                int coord = is_row_stage ? x : y;

                // Calculate butterfly indices (simple integer math)
                int group = coord / i.butterfly_size;
                int idx_in_group = coord % i.butterfly_size;
                int wing = idx_in_group / i.span;
                int idx_in_wing = idx_in_group % i.span;

                // Accumulate DFT sum
                float sum_real = 0.0;
                float sum_imag = 0.0;

                // Main DFT loop
                for (int j = 0; j < _Radix; j++)
                {
                    // Calculate input position
                    int input_pos = group * i.butterfly_size + j * i.span + idx_in_wing;

                    // Read input value
                    float in_real, in_imag;
                    if (is_row_stage)
                    {
                        float2 input_uv = float2((input_pos + 0.5) / (float)_N, i.uv.y);
                        float4 input_tex = _MainTex.SampleLevel(point_clamp_s, input_uv, 0);
                        if (_Stage == 0) {
                            // Assume that input is grayscale and real-valued.
                            in_real = input_tex.x;
                            in_imag = 0;
                        } else {
                            in_real = input_tex.x;
                            in_imag = input_tex.y;
                        }
                    }
                    else
                    {
                        float xuv = (x + 0.5) / _N;
                        float2 input_uv = float2(xuv, (input_pos + 0.5) / (float)_N);
                        float4 input_tex = _MainTex.SampleLevel(point_clamp_s, input_uv, 0);
                        in_real = input_tex.x;
                        in_imag = input_tex.y;
                    }

                    // Read DFT coefficient
                    float2 coeff = _Inverse > 0.5 ? IDFT_MATRIX[wing][j] : DFT_MATRIX[wing][j];
                    float coeff_real = coeff.x;
                    float coeff_imag = coeff.y;

                    // Complex multiply-accumulate
                    sum_real += coeff_real * in_real - coeff_imag * in_imag;
                    sum_imag += coeff_real * in_imag + coeff_imag * in_real;
                }

                // Apply stage twiddle if needed
                float out_real, out_imag;
                if (wing > 0 && idx_in_wing > 0)
                {
                    int twiddle_idx = wing * idx_in_wing;
                    float2 tw;

                    if (_Stage % 2 == 0) {
                        tw = _Inverse > 0.5 ? STAGE0_TWIDDLES_INV[twiddle_idx] : STAGE0_TWIDDLES[twiddle_idx];
                    } else {
                        tw = _Inverse > 0.5 ? STAGE1_TWIDDLES_INV[twiddle_idx] : STAGE1_TWIDDLES[twiddle_idx];
                    }

                    float tw_real = tw.x;
                    float tw_imag = tw.y;

                    // Output = twiddle * sum
                    out_real = tw_real * sum_real - tw_imag * sum_imag;
                    out_imag = tw_real * sum_imag + tw_imag * sum_real;
                }
                else
                {
                    out_real = sum_real;
                    out_imag = sum_imag;
                }

                // Handle final stage of inverse FFT
                if (_Inverse > 0.5 && _Stage == i.num_stages_per_dim * 2 - 1) {
                    float normalized = out_real / (_N * _N);
                    return float4(normalized, normalized, normalized, 1);
                }

                // Pack complex result into RGBA
                return float4(out_real, out_imag, 0, 1);
            }
            ENDCG
        }
    }
}

