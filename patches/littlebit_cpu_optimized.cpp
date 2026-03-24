#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace littlebit_cpu_ops {

namespace {

// ============================================================================
// Scalar fallbacks
// ============================================================================

inline int32_t dot_int8xsign_word_scalar(const int8_t * input, uint32_t bits, int64_t cols_this_word) {
    int32_t acc = 0;
    for (int64_t bit = 0; bit < cols_this_word; ++bit) {
        const int32_t value = static_cast<int32_t>(input[bit]);
        const bool is_neg = ((bits >> bit) & 1U) != 0;
        acc += is_neg ? -value : value;
    }
    return acc;
}

inline int32_t dot_int8xsign_row_scalar(
    const int8_t * input_row,
    const int32_t * packed_weight_row,
    int64_t n_cols) {
    int32_t acc = 0;
    const int64_t full_words = n_cols / 32;
    const int64_t tail_cols = n_cols % 32;

    for (int64_t word_idx = 0; word_idx < full_words; ++word_idx) {
        acc += dot_int8xsign_word_scalar(
            input_row + word_idx * 32,
            static_cast<uint32_t>(packed_weight_row[word_idx]),
            32
        );
    }

    if (tail_cols > 0) {
        acc += dot_int8xsign_word_scalar(
            input_row + full_words * 32,
            static_cast<uint32_t>(packed_weight_row[full_words]),
            tail_cols
        );
    }

    return acc;
}

inline int64_t dot_int32xsign_word_scalar(const int32_t * input, uint32_t bits, int64_t cols_this_word) {
    int64_t acc = 0;
    for (int64_t bit = 0; bit < cols_this_word; ++bit) {
        const int64_t value = static_cast<int64_t>(input[bit]);
        const bool is_neg = ((bits >> bit) & 1U) != 0;
        acc += is_neg ? -value : value;
    }
    return acc;
}

inline int64_t dot_int32xsign_row_scalar(
    const int32_t * input_row,
    const int32_t * packed_weight_row,
    int64_t n_cols) {
    int64_t acc = 0;
    const int64_t full_words = n_cols / 32;
    const int64_t tail_cols = n_cols % 32;

    for (int64_t word_idx = 0; word_idx < full_words; ++word_idx) {
        const uint32_t bits = static_cast<uint32_t>(packed_weight_row[word_idx]);
        acc += dot_int32xsign_word_scalar(input_row + word_idx * 32, bits, 32);
    }

    if (tail_cols > 0) {
        acc += dot_int32xsign_word_scalar(
            input_row + full_words * 32,
            static_cast<uint32_t>(packed_weight_row[full_words]),
            tail_cols
        );
    }

    return acc;
}

// ============================================================================
// AVX2 optimized paths
// ============================================================================

#ifdef __AVX2__

// --- Core: expand 32 packed sign bits to 32 sign bytes {-1, +1} ---
// Uses bit-manipulation instead of LUT. Inspired by llama.cpp's approach.
// bit=1 → -1 (negative), bit=0 → +1 (positive)
inline __m256i bits_to_sign_bytes(__m256i vbits, const __m256i & byte_sel, const __m256i & bit_mask) {
    // Shuffle to place the correct source byte for each of 32 lanes
    vbits = _mm256_shuffle_epi8(vbits, byte_sel);
    // Isolate one bit per lane
    __m256i is_set = _mm256_cmpeq_epi8(
        _mm256_and_si256(vbits, bit_mask), bit_mask);
    // is_set = 0xFF where bit=1, 0x00 where bit=0
    // We want: bit=1 → -1, bit=0 → +1
    // -1 = 0xFF in int8, so: result = is_set | 1 gives -1 or +1
    return _mm256_or_si256(is_set, _mm256_set1_epi8(1));
}

inline __m256i make_bits_broadcast(uint32_t packed_bits) {
    return _mm256_set1_epi32(static_cast<int32_t>(packed_bits));
}

// Pre-computed constants for bits_to_sign_bytes
static const __m256i k_byte_sel = _mm256_setr_epi8(
    0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
    2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
static const __m256i k_bit_mask = _mm256_setr_epi8(
    1,2,4,8,16,32,64,-128, 1,2,4,8,16,32,64,-128,
    1,2,4,8,16,32,64,-128, 1,2,4,8,16,32,64,-128);

inline int32_t hsum_epi32(__m256i value) {
    __m128i sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(value),
        _mm256_extracti128_si256(value, 1)
    );
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    return _mm_cvtsi128_si32(sum128);
}

// --- Optimized int8 × sign dot product for 32 elements ---
// Uses _mm256_sign_epi8: sign_epi8(a, b) = a * sign(b)
// Then sums via maddubs + madd pattern (fastest on x86)
inline int32_t dot_int8xsign_word32_avx2(const int8_t * input, uint32_t packed_bits) {
    const __m256i x_bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input));
    const __m256i signs = bits_to_sign_bytes(
        make_bits_broadcast(packed_bits), k_byte_sel, k_bit_mask);

    // Apply sign: effectively x * sign(weight)
    const __m256i signed_x = _mm256_sign_epi8(x_bytes, signs);

    // Sum all 32 int8 values: widen to int16 pairs, then to int32
    // maddubs treats first arg as unsigned, so we use abs values + manual sign
    // Instead, use the madd_epi16 with ones trick:
    // First widen int8→int16 by adding pairs, then int16→int32
    const __m256i ones = _mm256_set1_epi8(1);
    // maddubs(ones_unsigned, signed_x) = sum of pairs as int16
    const __m256i sum16 = _mm256_maddubs_epi16(
        _mm256_abs_epi8(signed_x),
        _mm256_sign_epi8(ones, signed_x));  // keep the sign
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i sum32 = _mm256_madd_epi16(sum16, ones16);
    return hsum_epi32(sum32);
}

// --- 4-word unrolled: process 128 elements at once ---
inline int32_t dot_int8xsign_4words_avx2(
    const int8_t * input,
    const int32_t * packed_words) {
    __m256i acc32 = _mm256_setzero_si256();
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i ones8 = _mm256_set1_epi8(1);

    for (int w = 0; w < 4; ++w) {
        const __m256i x_bytes = _mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(input + w * 32));
        const __m256i signs = bits_to_sign_bytes(
            make_bits_broadcast(static_cast<uint32_t>(packed_words[w])),
            k_byte_sel, k_bit_mask);
        const __m256i signed_x = _mm256_sign_epi8(x_bytes, signs);

        const __m256i sum16 = _mm256_maddubs_epi16(
            _mm256_abs_epi8(signed_x),
            _mm256_sign_epi8(ones8, signed_x));
        acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(sum16, ones16));
    }

    return hsum_epi32(acc32);
}

inline int32_t dot_int8xsign_row_avx2(
    const int8_t * input_row,
    const int32_t * packed_weight_row,
    int64_t n_cols) {
    int32_t acc = 0;
    const int64_t full_words = n_cols / 32;
    const int64_t tail_cols = n_cols % 32;

    // Process 4 words (128 elements) at a time
    int64_t word_idx = 0;
    const int64_t full_4words = (full_words / 4) * 4;
    for (; word_idx < full_4words; word_idx += 4) {
        acc += dot_int8xsign_4words_avx2(
            input_row + word_idx * 32,
            packed_weight_row + word_idx);
    }

    // Remaining full words (1-3)
    for (; word_idx < full_words; ++word_idx) {
        acc += dot_int8xsign_word32_avx2(
            input_row + word_idx * 32,
            static_cast<uint32_t>(packed_weight_row[word_idx])
        );
    }

    if (tail_cols > 0) {
        acc += dot_int8xsign_word_scalar(
            input_row + full_words * 32,
            static_cast<uint32_t>(packed_weight_row[full_words]),
            tail_cols
        );
    }

    return acc;
}

// --- int32 × sign dot (for stage 2) ---
inline int64_t dot_int32xsign_word8_avx2(const int32_t * input, const __m256i & signs_i32) {
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input));
    const __m256i prod = _mm256_mullo_epi32(x, signs_i32);
    return static_cast<int64_t>(hsum_epi32(prod));
}

inline int64_t dot_int32xsign_row_avx2(
    const int32_t * input_row,
    const int32_t * packed_weight_row,
    int64_t n_cols) {
    int64_t acc = 0;
    const int64_t full_words = n_cols / 32;
    const int64_t tail_cols = n_cols % 32;

    for (int64_t word_idx = 0; word_idx < full_words; ++word_idx) {
        const uint32_t bits = static_cast<uint32_t>(packed_weight_row[word_idx]);
        const __m256i vbits = make_bits_broadcast(bits);
        const __m256i signs = bits_to_sign_bytes(vbits, k_byte_sel, k_bit_mask);

        // Process 8 int32 values at a time (need to widen sign bytes to int32)
        const int64_t base = word_idx * 32;
        for (int sub = 0; sub < 4; ++sub) {
            // Extract 8 sign bytes, widen to int32
            __m128i sign_slice;
            if (sub == 0) sign_slice = _mm256_castsi256_si128(signs);
            else if (sub == 1) sign_slice = _mm_srli_si128(_mm256_castsi256_si128(signs), 8);
            else if (sub == 2) sign_slice = _mm256_extracti128_si256(signs, 1);
            else sign_slice = _mm_srli_si128(_mm256_extracti128_si256(signs, 1), 8);

            // Only take lower 8 bytes → widen to 8 int32
            const __m256i signs_i32 = _mm256_cvtepi8_epi32(sign_slice);
            const __m256i x = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(input_row + base + sub * 8));
            const __m256i prod = _mm256_mullo_epi32(x, signs_i32);
            acc += static_cast<int64_t>(hsum_epi32(prod));
        }
    }

    if (tail_cols > 0) {
        acc += dot_int32xsign_word_scalar(
            input_row + full_words * 32,
            static_cast<uint32_t>(packed_weight_row[full_words]),
            tail_cols
        );
    }

    return acc;
}

// --- Quantization helpers (AVX2) ---

inline float reduce_max_lanes(__m256 max_vec) {
    alignas(32) float lanes[8];
    _mm256_storeu_ps(lanes, max_vec);
    float max_abs = 0.0f;
    for (float lane : lanes) {
        max_abs = std::max(max_abs, lane);
    }
    return max_abs;
}

inline float max_abs_row_avx2(const float * input_row, int64_t cols) {
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 max_vec = _mm256_setzero_ps();
    int64_t col = 0;
    for (; col + 8 <= cols; col += 8) {
        const __m256 value = _mm256_loadu_ps(input_row + col);
        max_vec = _mm256_max_ps(max_vec, _mm256_and_ps(value, abs_mask));
    }

    float max_abs = reduce_max_lanes(max_vec);
    for (; col < cols; ++col) {
        max_abs = std::max(max_abs, std::abs(input_row[col]));
    }
    return max_abs;
}

inline float max_abs_mul_row_avx2(const float * input_row, const float * mul_row, int64_t cols) {
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 max_vec = _mm256_setzero_ps();
    int64_t col = 0;
    for (; col + 8 <= cols; col += 8) {
        const __m256 input_vec = _mm256_loadu_ps(input_row + col);
        const __m256 mul_vec = _mm256_loadu_ps(mul_row + col);
        const __m256 value = _mm256_mul_ps(input_vec, mul_vec);
        max_vec = _mm256_max_ps(max_vec, _mm256_and_ps(value, abs_mask));
    }

    float max_abs = reduce_max_lanes(max_vec);
    for (; col < cols; ++col) {
        max_abs = std::max(max_abs, std::abs(input_row[col] * mul_row[col]));
    }
    return max_abs;
}

inline void store_quantized_8(__m256i ints, int8_t * quantized_row) {
    const __m128i ints_lo = _mm256_castsi256_si128(ints);
    const __m128i ints_hi = _mm256_extracti128_si256(ints, 1);
    const __m128i packed_16 = _mm_packs_epi32(ints_lo, ints_hi);
    const __m128i packed_8 = _mm_packs_epi16(packed_16, packed_16);
    _mm_storel_epi64(reinterpret_cast<__m128i *>(quantized_row), packed_8);
}

inline void quantize_row_avx2(
    const float * input_row,
    int64_t cols,
    float scale,
    int8_t * quantized_row) {
    const __m256 scale_vec = _mm256_set1_ps(scale);
    const __m256 min_vec = _mm256_set1_ps(-128.0f);
    const __m256 max_vec = _mm256_set1_ps(127.0f);

    int64_t col = 0;
    for (; col + 8 <= cols; col += 8) {
        const __m256 input_vec = _mm256_loadu_ps(input_row + col);
        const __m256 scaled = _mm256_mul_ps(input_vec, scale_vec);
        const __m256 clamped = _mm256_max_ps(min_vec, _mm256_min_ps(max_vec, scaled));
        const __m256i rounded = _mm256_cvtps_epi32(clamped);
        store_quantized_8(rounded, quantized_row + col);
    }

    for (; col < cols; ++col) {
        const float scaled = std::nearbyint(input_row[col] * scale);
        const float clamped = std::max(-128.0f, std::min(127.0f, scaled));
        quantized_row[col] = static_cast<int8_t>(clamped);
    }
}

inline void quantize_mul_row_avx2(
    const float * input_row,
    const float * mul_row,
    int64_t cols,
    float scale,
    int8_t * quantized_row) {
    const __m256 scale_vec = _mm256_set1_ps(scale);
    const __m256 min_vec = _mm256_set1_ps(-128.0f);
    const __m256 max_vec = _mm256_set1_ps(127.0f);

    int64_t col = 0;
    for (; col + 8 <= cols; col += 8) {
        const __m256 input_vec = _mm256_loadu_ps(input_row + col);
        const __m256 mul_vec = _mm256_loadu_ps(mul_row + col);
        const __m256 product = _mm256_mul_ps(input_vec, mul_vec);
        const __m256 scaled = _mm256_mul_ps(product, scale_vec);
        const __m256 clamped = _mm256_max_ps(min_vec, _mm256_min_ps(max_vec, scaled));
        const __m256i rounded = _mm256_cvtps_epi32(clamped);
        store_quantized_8(rounded, quantized_row + col);
    }

    for (; col < cols; ++col) {
        const float scaled = std::nearbyint(input_row[col] * mul_row[col] * scale);
        const float clamped = std::max(-128.0f, std::min(127.0f, scaled));
        quantized_row[col] = static_cast<int8_t>(clamped);
    }
}
#endif  // __AVX2__

// ============================================================================
// Dispatch wrappers
// ============================================================================

inline float quantize_mul_row_to_int8(
    const float * input_row,
    const float * mul_row,
    int64_t cols,
    int8_t * quantized_row) {
    float max_abs = 0.0f;
#ifdef __AVX2__
    max_abs = max_abs_mul_row_avx2(input_row, mul_row, cols);
#else
    for (int64_t col = 0; col < cols; ++col) {
        const float value = input_row[col] * mul_row[col];
        max_abs = std::max(max_abs, std::abs(value));
    }
#endif

    const float scale = 127.0f / std::max(max_abs, 1e-5f);
#ifdef __AVX2__
    quantize_mul_row_avx2(input_row, mul_row, cols, scale, quantized_row);
#else
    for (int64_t col = 0; col < cols; ++col) {
        const float scaled = std::nearbyint(input_row[col] * mul_row[col] * scale);
        const float clamped = std::max(-128.0f, std::min(127.0f, scaled));
        quantized_row[col] = static_cast<int8_t>(clamped);
    }
#endif
    return scale;
}

inline float quantize_row_to_int8(
    const float * input_row,
    int64_t cols,
    int8_t * quantized_row) {
    float max_abs = 0.0f;
#ifdef __AVX2__
    max_abs = max_abs_row_avx2(input_row, cols);
#else
    for (int64_t col = 0; col < cols; ++col) {
        max_abs = std::max(max_abs, std::abs(input_row[col]));
    }
#endif

    const float scale = 127.0f / std::max(max_abs, 1e-5f);
#ifdef __AVX2__
    quantize_row_avx2(input_row, cols, scale, quantized_row);
#else
    for (int64_t col = 0; col < cols; ++col) {
        const float scaled = std::nearbyint(input_row[col] * scale);
        const float clamped = std::max(-128.0f, std::min(127.0f, scaled));
        quantized_row[col] = static_cast<int8_t>(clamped);
    }
#endif
    return scale;
}

template <typename scalar_t>
inline float quantize_input_row_to_int8(
    const scalar_t * input_row,
    int64_t cols,
    int8_t * quantized_row) {
    float max_abs = 0.0f;
    for (int64_t col = 0; col < cols; ++col) {
        const float value = static_cast<float>(input_row[col]);
        max_abs = std::max(max_abs, std::abs(value));
    }

    const float scale = 127.0f / std::max(max_abs, 1e-5f);
    for (int64_t col = 0; col < cols; ++col) {
        const float scaled = std::nearbyint(static_cast<float>(input_row[col]) * scale);
        const float clamped = std::max(-128.0f, std::min(127.0f, scaled));
        quantized_row[col] = static_cast<int8_t>(clamped);
    }
    return scale;
}

template <>
inline float quantize_input_row_to_int8<float>(
    const float * input_row,
    int64_t cols,
    int8_t * quantized_row) {
    return quantize_row_to_int8(input_row, cols, quantized_row);
}

inline int32_t dot_int8xsign_row(
    const int8_t * input_row,
    const int32_t * packed_weight_row,
    int64_t n_cols) {
#ifdef __AVX2__
    return dot_int8xsign_row_avx2(input_row, packed_weight_row, n_cols);
#else
    return dot_int8xsign_row_scalar(input_row, packed_weight_row, n_cols);
#endif
}

inline int64_t dot_int32xsign_row(
    const int32_t * input_row,
    const int32_t * packed_weight_row,
    int64_t n_cols) {
#ifdef __AVX2__
    return dot_int32xsign_row_avx2(input_row, packed_weight_row, n_cols);
#else
    return dot_int32xsign_row_scalar(input_row, packed_weight_row, n_cols);
#endif
}

template <typename Fn>
inline void parallel_for_maybe_serial(int64_t begin, int64_t end, int64_t serial_limit, Fn && fn) {
    if (end <= begin) {
        return;
    }
    if ((end - begin) <= serial_limit || at::get_num_threads() == 1) {
        fn(begin, end);
        return;
    }
    at::parallel_for(begin, end, 0, std::forward<Fn>(fn));
}

}  // namespace

void check_cpu_tensor(const at::Tensor & tensor, const char * name) {
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_float32_tensor(const at::Tensor & tensor, const char * name) {
    check_cpu_tensor(tensor, name);
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
}

std::tuple<at::Tensor, at::Tensor> quantize_per_row_int8_cpu(const at::Tensor & input) {
    check_cpu_tensor(input, "input");
    TORCH_CHECK(input.dim() == 2, "input must have shape [M, K]");
    TORCH_CHECK(input.is_floating_point(), "input must be floating point");

    const auto rows = input.size(0);
    const auto cols = input.size(1);

    auto q = at::empty({rows, cols}, input.options().dtype(at::kChar));
    auto scale = at::empty({rows, 1}, input.options().dtype(at::kFloat));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "littlebit_cpu_quantize_per_row_int8",
        [&] {
            const auto * input_ptr = input.data_ptr<scalar_t>();
            auto * q_ptr = q.data_ptr<int8_t>();
            auto * scale_ptr = scale.data_ptr<float>();

            at::parallel_for(0, rows, 0, [&](int64_t begin, int64_t end) {
                for (int64_t row = begin; row < end; ++row) {
                    const auto * row_ptr = input_ptr + row * cols;
                    auto * q_row_ptr = q_ptr + row * cols;
                    scale_ptr[row] = quantize_input_row_to_int8(row_ptr, cols, q_row_ptr);
                }
            });
        });

    return std::make_tuple(q, scale);
}

at::Tensor gemv_int8xsign_cpu(
    const at::Tensor & input,
    const at::Tensor & packed_weight,
    int64_t n_cols) {
    check_cpu_tensor(input, "input");
    check_cpu_tensor(packed_weight, "packed_weight");

    TORCH_CHECK(input.scalar_type() == at::kChar, "input must be int8");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(input.dim() == 2, "input must have shape [M, K]");
    TORCH_CHECK(packed_weight.dim() == 2, "packed_weight must have shape [N, ceil(K/32)]");
    TORCH_CHECK(input.size(1) == n_cols, "input K must match n_cols");

    const int64_t expected_words = (n_cols + 31) / 32;
    TORCH_CHECK(
        packed_weight.size(1) == expected_words,
        "packed_weight second dimension must equal ceil(n_cols / 32)"
    );

    const auto m = input.size(0);
    const auto n = packed_weight.size(0);
    const auto words_per_row = packed_weight.size(1);

    auto output = at::zeros({m, n}, input.options().dtype(at::kInt));

    const auto * input_ptr = input.data_ptr<int8_t>();
    const auto * weight_ptr = packed_weight.data_ptr<int32_t>();
    auto * out_ptr = output.data_ptr<int32_t>();

    at::parallel_for(0, m * n, 0, [&](int64_t begin, int64_t end) {
        for (int64_t index = begin; index < end; ++index) {
            const int64_t sample = index / n;
            const int64_t out_row = index % n;

            const auto * input_row = input_ptr + sample * n_cols;
            const auto * weight_row = weight_ptr + out_row * words_per_row;

            out_ptr[sample * n + out_row] = dot_int8xsign_row(
                input_row, weight_row, n_cols);
        }
    });

    return output;
}

at::Tensor littlebit_linear_cpu(
    const at::Tensor & input,
    const at::Tensor & v2,
    const at::Tensor & v_sign,
    int64_t v_cols,
    const at::Tensor & mid,
    const at::Tensor & u_sign,
    int64_t u_cols,
    const at::Tensor & u1) {
    check_float32_tensor(input, "input");
    check_float32_tensor(v2, "v2");
    check_cpu_tensor(v_sign, "v_sign");
    check_float32_tensor(mid, "mid");
    check_cpu_tensor(u_sign, "u_sign");
    check_float32_tensor(u1, "u1");

    TORCH_CHECK(v_sign.scalar_type() == at::kInt, "v_sign must be int32");
    TORCH_CHECK(u_sign.scalar_type() == at::kInt, "u_sign must be int32");
    TORCH_CHECK(input.dim() == 2, "input must have shape [M, K]");
    TORCH_CHECK(v2.dim() == 2 && v2.size(0) == 1, "v2 must have shape [1, K]");
    TORCH_CHECK(mid.dim() == 2 && mid.size(0) == 1, "mid must have shape [1, R]");
    TORCH_CHECK(u1.dim() == 2 && u1.size(0) == 1, "u1 must have shape [1, N]");
    TORCH_CHECK(input.size(1) == v_cols, "input K must match v_cols");
    TORCH_CHECK(v2.size(1) == v_cols, "v2 second dimension must match v_cols");
    TORCH_CHECK(v_sign.dim() == 2, "v_sign must have shape [R, ceil(K/32)]");
    TORCH_CHECK(u_sign.dim() == 2, "u_sign must have shape [N, ceil(R/32)]");
    TORCH_CHECK(v_sign.size(0) == mid.size(1), "mid rank must match v_sign rows");
    TORCH_CHECK(u_sign.size(0) == u1.size(1), "u1 out_features must match u_sign rows");
    TORCH_CHECK(mid.size(1) == u_cols, "mid rank must match u_cols");
    TORCH_CHECK(
        v_sign.size(1) == (v_cols + 31) / 32,
        "v_sign second dimension must equal ceil(v_cols / 32)"
    );
    TORCH_CHECK(
        u_sign.size(1) == (u_cols + 31) / 32,
        "u_sign second dimension must equal ceil(u_cols / 32)"
    );

    const auto rows = input.size(0);
    const auto rank = v_sign.size(0);
    const auto out_features = u_sign.size(0);
    const auto v_words = v_sign.size(1);
    const auto u_words = u_sign.size(1);

    auto output = at::empty({rows, out_features}, input.options().dtype(at::kFloat));

    const auto * input_ptr = input.data_ptr<float>();
    const auto * v2_ptr = v2.data_ptr<float>();
    const auto * v_sign_ptr = v_sign.data_ptr<int32_t>();
    const auto * mid_ptr = mid.data_ptr<float>();
    const auto * u_sign_ptr = u_sign.data_ptr<int32_t>();
    const auto * u1_ptr = u1.data_ptr<float>();
    auto * out_ptr = output.data_ptr<float>();

    struct LittleBitScratch {
        std::vector<int8_t> q1;
        std::vector<float> stage1;
        std::vector<int8_t> q2;
    };
    thread_local LittleBitScratch scratch;
    scratch.q1.resize(v_cols);
    scratch.stage1.resize(rank);
    scratch.q2.resize(rank);

    auto * q1_ptr = scratch.q1.data();
    auto * stage1_ptr = scratch.stage1.data();
    auto * q2_ptr = scratch.q2.data();
    for (int64_t row = 0; row < rows; ++row) {
        const auto * input_row = input_ptr + row * v_cols;
        auto * out_row = out_ptr + row * out_features;

        const float scale1 = quantize_mul_row_to_int8(input_row, v2_ptr, v_cols, q1_ptr);
        const float inv_scale1 = 1.0f / scale1;

        // Stage 1: V projection — rank is typically small (32), run serial
        parallel_for_maybe_serial(0, rank, 256, [&](int64_t begin, int64_t end) {
            for (int64_t rank_idx = begin; rank_idx < end; ++rank_idx) {
                const int32_t acc = dot_int8xsign_row(
                    q1_ptr,
                    v_sign_ptr + rank_idx * v_words,
                    v_cols
                );
                stage1_ptr[rank_idx] = static_cast<float>(acc) * inv_scale1 * mid_ptr[rank_idx];
            }
        });

        const float scale2 = quantize_row_to_int8(stage1_ptr, rank, q2_ptr);
        const float inv_scale2 = 1.0f / scale2;

        // Stage 2: U projection — out_features=4096, parallelize aggressively
        parallel_for_maybe_serial(0, out_features, 256, [&](int64_t begin, int64_t end) {
            for (int64_t out_idx = begin; out_idx < end; ++out_idx) {
                const int32_t acc = dot_int8xsign_row(
                    q2_ptr,
                    u_sign_ptr + out_idx * u_words,
                    u_cols
                );
                out_row[out_idx] = static_cast<float>(acc) * inv_scale2 * u1_ptr[out_idx];
            }
        });
    }

    return output;
}

}  // namespace littlebit_cpu_ops

TORCH_LIBRARY(littlebit_cpu_ops, m) {
    m.def("quantize_per_row_int8(Tensor input) -> (Tensor, Tensor)");
    m.def("gemv_int8xsign(Tensor input, Tensor packed_weight, int n_cols) -> Tensor");
    m.def(
        "littlebit_linear("
        "Tensor input, Tensor v2, Tensor v_sign, int v_cols, "
        "Tensor mid, Tensor u_sign, int u_cols, Tensor u1"
        ") -> Tensor"
    );
}

TORCH_LIBRARY_IMPL(littlebit_cpu_ops, CPU, m) {
    m.impl("quantize_per_row_int8", littlebit_cpu_ops::quantize_per_row_int8_cpu);
    m.impl("gemv_int8xsign", littlebit_cpu_ops::gemv_int8xsign_cpu);
    m.impl("littlebit_linear", littlebit_cpu_ops::littlebit_linear_cpu);
}
