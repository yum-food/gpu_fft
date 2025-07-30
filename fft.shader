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
            #define RADIX 16
            #define N 256
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

            float4 cmul_2x(float4 a, float2 b) {
                float4 r;
                r.xz = a.xz * b.x - a.yw * b.y;
                r.yw = a.xz * b.y + a.yw * b.x;
                return r;
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
                  float3 col = _MainTex.SampleLevel(point_clamp_s, i.uv, 0).rgb;
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

                // Calculate butterfly indices
                const int group = coord / i.butterfly_size;
                const int idx_in_group = coord % i.butterfly_size;
                const int wing = idx_in_group / i.span;
                const int idx_in_wing = idx_in_group % i.span;

                // Main DFT loop
                float4 sum = float4(0.0, 0.0, 0.0, 0.0);
                for (int j = 0; j < RADIX; j++)
                {
                    // Calculate input position
                    const int input_pos = group * i.butterfly_size + j * i.span + idx_in_wing;

                    // Read input value
                    float4 input_tex;
                    if (is_row_stage)
                    {
                        const float2 input_uv = float2((input_pos + 0.5) / (float)N, i.uv.y);
                        input_tex = _MainTex.SampleLevel(point_clamp_s, input_uv, 0);
                    }
                    else
                    {
                        const float2 input_uv = float2(i.uv.x, (input_pos + 0.5) / (float)N);
                        input_tex = _MainTex.SampleLevel(point_clamp_s, input_uv, 0);
                    }

                    // Read DFT coefficient
                    const float2 coeff = _Inverse ? IDFT_MATRIX[wing][j] : DFT_MATRIX[wing][j];

                    // Complex multiply-accumulate
                    sum += cmul_2x(input_tex, coeff);
                }

                // Apply stage twiddle if needed
                float4 out_val;
                if (wing > 0 && idx_in_wing > 0)
                {
                    const int twiddle_idx = wing * idx_in_wing;
                    float2 tw;

                    if (_Stage % 2 == 0) {
                        tw = _Inverse ? STAGE0_TWIDDLES_INV[twiddle_idx] : STAGE0_TWIDDLES[twiddle_idx];
                    } else {
                        tw = _Inverse ? STAGE1_TWIDDLES_INV[twiddle_idx] : STAGE1_TWIDDLES[twiddle_idx];
                    }

                    // Output = twiddle * sum
                    out_val = cmul_2x(sum, tw);
                }
                else
                {
                    out_val = sum;
                }
                if (_Inverse && _Stage == 3) {
                    out_val /= (_N * _N);
                }
                return out_val;
            }
            ENDCG
        }
    }
}

