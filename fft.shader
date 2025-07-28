Shader "yum_food/fft"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _N ("N", Int) = 256
        _Radix ("Radix", Int) = 16
        _Stage ("Stage", Int) = 0
        [Toggle] _PassThrough ("Pass Through", Float) = 0
        [Toggle] _LDS ("Temporal LDS", Float) = 0
        [Toggle] _Inverse ("Inverse FFT", Float) = 0
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
            #define GPU_FFT_RADIX16_SIZE256
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
                int num_stages_per_dim : TEXCOORD1;
                int span : TEXCOORD2;
                int butterfly_size : TEXCOORD3;
                int num_stages : TEXCOORD4;
            };

            texture2D _MainTex;
            SamplerState point_clamp_s;
            int _N;
            int _Radix;
            int _Stage;
            float _PassThrough;
            float _LDS;
            float _Inverse;

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

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;

                // Calculate num_stages_per_dim = log_radix(N)
                o.num_stages_per_dim = (int)(log(_N) / log(_Radix));

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
                int2 pixel_index = (int2)(i.uv * _N);
                float2 uv = (pixel_index + 0.5f) / _N;

                // If pass through is enabled, just return the input
                if (_PassThrough > 0.5)
                {
                  float3 col = _MainTex.SampleLevel(point_clamp_s, uv, 0).rgb;
                  if (_LDS > 0.5) {
                    col += PHI * _Time[0];
                    col = frac(col);
                  }
                  float lum = luminance(col);
                  return float4(lum, lum, lum, 1);
                }

                // Calculate pixel index from UV coordinates
                int x = pixel_index.x;
                int y = pixel_index.y;

                // Determine processing direction (row stage or column stage)
                bool is_row_stage = (_Stage < i.num_stages_per_dim);
                int coord = is_row_stage ? x : y;

                // Calculate butterfly indices
                int group = coord / i.butterfly_size;
                int idx_in_group = coord % i.butterfly_size;
                int wing = idx_in_group / i.span;
                int idx_in_wing = idx_in_group % i.span;

                // Accumulate DFT sum
                float2 sum = float2(0.0, 0.0);

                // Main DFT loop
                for (int j = 0; j < _Radix; j++)
                {
                    // Calculate input position
                    int input_pos = group * i.butterfly_size + j * i.span + idx_in_wing;

                    // Calculate UV for input texture read
                    float2 input_uv;
                    if (is_row_stage)
                    {
                        input_uv = float2((input_pos + 0.5) / (float)_N, i.uv.y);
                    }
                    else
                    {
                        float xuv = (x + 0.5) / _N;
                        input_uv = float2(xuv, (input_pos + 0.5) / (float)_N);
                    }

                    // Read input value
                    float4 input_tex = _MainTex.SampleLevel(point_clamp_s, input_uv, 0);
                    float2 input_val;
                    if (_Stage == 0) {
                      input_val.x = luminance(input_tex.xyz);
                      input_val.y = 0;
                    } else {
                      input_val.x = input_tex.x + input_tex.y;
                      input_val.y = input_tex.z + input_tex.w;
                    }

                    // Read DFT coefficient from the table (use inverse matrix if _Inverse is set)
                    float2 coeff = _Inverse > 0.5 ? IDFT_MATRIX[wing][j] : DFT_MATRIX[wing][j];

                    // Complex multiply-accumulate
                    sum.x += coeff.x * input_val.x - coeff.y * input_val.y;
                    sum.y += coeff.x * input_val.y + coeff.y * input_val.x;
                }

                // Apply stage twiddle if needed
                if (wing > 0 && idx_in_wing > 0)
                {
                    int twiddle_idx = wing * idx_in_wing;
                    float2 tw = _Inverse > 0.5 ? STAGE_TWIDDLES_INV[twiddle_idx] : STAGE_TWIDDLES[twiddle_idx];

                    // Output = twiddle * sum
                    float2 output;
                    output.x = tw.x * sum.x - tw.y * sum.y;
                    output.y = tw.x * sum.y + tw.y * sum.x;
                    sum = output;
                }

                // Pack complex result into RGBA.
                float real_part = sum.x;
                float imag_part = sum.y;

                if (_Inverse > 0.5 && _Stage == i.num_stages_per_dim * 2 - 1) {
                  // Last stage of IFFT is just back to the original real-valued signal.
                  real_part /= _N * i.num_stages_per_dim;
                  return float4(real_part, real_part, real_part, 1);
                }

                // Split into 2 parts.
                float real_big   = floor(real_part);
                float real_small = real_part - real_big;
                float imag_big   = floor(imag_part);
                float imag_small = imag_part - imag_big;

                return float4(real_big, real_small, imag_big, imag_small);
            }
            ENDCG
        }
    }
}

