Shader "yum_food/fft"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _N ("N", Int) = 256
        _Radix ("Radix", Int) = 16
        _Stage ("Stage", Int) = 0
        [Toggle] _PassThrough ("Pass Through", Float) = 0
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
            };

            texture2D _MainTex;
            float4 _MainTex_ST;
            SamplerState point_repeat_s;
            int _N;
            int _Radix;
            int _Stage;
            float _PassThrough;

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
                o.num_stages_per_dim = (int)(log(_N) / log(_Radix) + 0.5);

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
                // If pass through is enabled, just return the input
                if (_PassThrough > 0.5)
                {
                  return _MainTex.SampleLevel(point_repeat_s, i.uv, 0);
                }
                const float n2 = _N * _N;

                // Calculate pixel index from UV coordinates
                int2 pixel_index = int2(floor(i.uv * _N));
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
                        input_uv = float2(i.uv.x, (input_pos + 0.5) / (float)_N);
                    }

                    // Read input value
                    float4 input_tex = _MainTex.SampleLevel(point_repeat_s, input_uv, 0);
                    float2 input_val;
                    if (_Stage == 0) {
                      input_val.x = luminance(input_tex.xyz);
                      input_val.y = 0;
                    } else {
                      // Remap onto [-1, 1]
                      input_tex = input_tex * 2.0f - 1.0f;
                      input_val.x = input_tex.x * n2 + input_tex.y;
                      input_val.y = input_tex.z * n2 + input_tex.w;
                    }

                    // Read DFT coefficient from the table
                    float2 coeff = DFT_MATRIX[wing][j];

                    // Complex multiply-accumulate
                    sum.x += coeff.x * input_val.x - coeff.y * input_val.y;
                    sum.y += coeff.x * input_val.y + coeff.y * input_val.x;
                }

                // Apply stage twiddle if needed
                if (wing > 0 && idx_in_wing > 0)
                {
                    int twiddle_idx = wing * idx_in_wing;
                    float2 tw = STAGE_TWIDDLES[twiddle_idx];

                    // Output = twiddle * sum
                    float2 output;
                    output.x = tw.x * sum.x - tw.y * sum.y;
                    output.y = tw.x * sum.y + tw.y * sum.x;
                    sum = output;
                }

                // Pack complex result into RGBA.
                float real_part = sum.x;
                float imag_part = sum.y;

                // Split into 2 parts.
                float real_big   = floor(real_part);
                float real_small = real_part - real_big;
                float imag_big   = floor(imag_part);
                float imag_small = imag_part - imag_big;

                // Compress onto [-1,1].
                // For an N*N FFT, the maximum value is N^2 and the min value
                // is -N^2 / 2.
                real_big /= n2;
                imag_big /= n2;

                // Map onto [0, 1].
                real_big   = (real_big + 1.0f) * 0.5f;
                real_small = (real_small + 1.0f) * 0.5f;
                imag_big   = (imag_big + 1.0f) * 0.5f;
                imag_small = (imag_small + 1.0f) * 0.5f;

                return float4(real_big, real_small, imag_big, imag_small);
            }
            ENDCG
        }
    }
}

