// The pitch stage's AVX loops, out of line so they are defined once, not in every user's header.
#include "PitchShifter.hpp"
#include <cerrno>
#include <cfenv>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #ifdef _MSC_VER
        #include <intrin.h>
        #define PITCH_AVX_TARGET
    #else
        #include <cpuid.h>
        #define PITCH_AVX_TARGET __attribute__((target("avx,sse4.2")))
    #endif

namespace pitch_simd {
// AVX in the CPU (CPUID 1, ECX bit 28) and its 256-bit registers saved by the OS: OSXSAVE (bit 27)
// plus the SSE and AVX state bits (1 and 2) of XCR0.
bool available() {
    static bool const supported = [] {
    #ifdef _MSC_VER
        int info[4];
        __cpuid(info, 1);
        unsigned const features = unsigned(info[2]);
    #else
        unsigned a = 0, b = 0, features = 0, d = 0;
        if (!__get_cpuid(1, &a, &b, &features, &d)) return false;
    #endif
        if (!(features & (1u << 27)) || !(features & (1u << 28))) return false;
    #ifdef _MSC_VER
        unsigned long long const enabled = _xgetbv(0);
    #else
        unsigned low = 0, high = 0;
        __asm__("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
        unsigned long long const enabled = low | (static_cast<unsigned long long>(high) << 32);
    #endif
        return (enabled & 6) == 6;
    }();
    return supported;
}

bool exactFloatMath() {
    #ifdef _MSC_VER
    static bool const exact = [] {
        if (!available()) return false;
        struct Restore {
            std::fenv_t environment;
            int const error = errno;
            Restore() {
                std::feholdexcept(&environment);
            }
            ~Restore() {
                std::fesetenv(&environment);
                errno = error;
            }
        } const restore;
        constexpr size_t count = 1 << 16;
        std::vector<float> re(count), im(count), angle(count), magnitude(count);
        float const corners[][2] = {
            {0.f, 0.f},
            {-0.f, 0.f},
            {0.f, -0.f},
            {-0.f, -0.f},
            {1.f, 0.f},
            {-1.f, 0.f},
            {0.f, 1.f},
            {0.f, -1.f},
            {1.f, 1.f},
            {-1.f, -1.f},
            {1.f, 0x1p-13f},
            {1.f, 0x1.fffffep-14f},
            {-1.f, 0x1p-13f},
            {1.f, .0625f},
            {1.f, 0x1.000002p-4f},
            {1.f, 0x1.fffffep-5f},
            {0x1p-149f, 1.f},
            {1.f, 0x1p-149f},
            {0x1p-126f, 0x1p-126f},
            {3.4e38f, -3.4e38f},
        };
        uint64_t state = 0x9e3779b97f4a7c15u;
        auto next = [&] {
            for (;;) {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                uint32_t const bits = uint32_t(state >> 32);
                float value;
                std::memcpy(&value, &bits, sizeof(value));
                if (std::isfinite(value)) return value;
            }
        };
        for (size_t k = 0; k < count; ++k) {
            bool const corner = k < std::size(corners);
            re[k] = corner ? corners[k][0] : next();
            im[k] = corner ? corners[k][1] : k % 7 == 0 ? re[k] * float(int(k % 3) - 1) : next();
        }
        arguments(re.data(), im.data(), angle.data(), count);
        magnitudes(re.data(), im.data(), magnitude.data(), count);
        for (size_t k = 0; k < count; ++k) {
            std::complex<float> const bin(re[k], im[k]);
            if (angle[k] != std::arg(bin) || magnitude[k] != std::abs(bin)) return false;
        }
        std::vector<float> phase(count), polarRe(count), polarIm(count);
        float const turns[] = {
            6.2831855f,
            -6.2831855f,
            .785398185f,
            .785398245f,
            1.57079637f,
            3.14159274f,
            4.71238899f,
            .0078125f,
            .00781249953f,
            1.52587891e-05f,
            1.52587882e-05f,
            -.0078125f,
            0.f,
            -0.f,
        };
        for (size_t k = 0; k < count; ++k)
            phase[k] = k < std::size(turns) ?
                turns[k] :
                float(std::fmod(double(angle[k]) * double(k % 5), 6.283185307179586));
        for (size_t k = 0; k < std::size(turns); ++k)
            magnitude[k] = 1.f;
        polar(magnitude.data(), phase.data(), polarRe.data(), polarIm.data(), count);
        for (size_t k = 0; k < count; ++k) {
            auto const bin =
                magnitude[k] == 0 ? std::complex<float>() : std::polar(magnitude[k], phase[k]);
            if (polarRe[k] != bin.real() || polarIm[k] != bin.imag()) return false;
        }
        return true;
    }();
    return exact;
    #else
    return false;
    #endif
}

PITCH_AVX_TARGET void butterflies(
    float* lowRe, float* lowIm, float* highRe, float* highIm, size_t half, float const* cosines,
    float const* sines
) {
    for (size_t k = 0; k < half; k += 8) {
        __m256 const hr = _mm256_loadu_ps(highRe + k), hi = _mm256_loadu_ps(highIm + k);
        __m256 const cosine = _mm256_loadu_ps(cosines + k), sine = _mm256_loadu_ps(sines + k);
        __m256 const br = _mm256_sub_ps(_mm256_mul_ps(hr, cosine), _mm256_mul_ps(hi, sine));
        __m256 const bi = _mm256_add_ps(_mm256_mul_ps(hr, sine), _mm256_mul_ps(hi, cosine));
        __m256 const lr = _mm256_loadu_ps(lowRe + k), li = _mm256_loadu_ps(lowIm + k);
        _mm256_storeu_ps(highRe + k, _mm256_sub_ps(lr, br));
        _mm256_storeu_ps(highIm + k, _mm256_sub_ps(li, bi));
        _mm256_storeu_ps(lowRe + k, _mm256_add_ps(lr, br));
        _mm256_storeu_ps(lowIm + k, _mm256_add_ps(li, bi));
    }
    _mm256_zeroupper(); // the caller's SSE code runs next
}

PITCH_AVX_TARGET void magnitudes(float const* re, float const* im, float* out, size_t count) {
    size_t k = 0;
    for (; k + 4 <= count; k += 4) {
        __m256d const real = _mm256_cvtps_pd(_mm_loadu_ps(re + k));
        __m256d const imaginary = _mm256_cvtps_pd(_mm_loadu_ps(im + k));
        __m256d const squares =
            _mm256_add_pd(_mm256_mul_pd(real, real), _mm256_mul_pd(imaginary, imaginary));
        _mm_storeu_ps(out + k, _mm256_cvtpd_ps(_mm256_sqrt_pd(squares)));
    }
    for (; k < count; ++k)
        out[k] = std::abs(std::complex<float>(re[k], im[k]));
    _mm256_zeroupper();
}

namespace {
constexpr double piOver4 = float(3.14159265358979323846 / 4);
constexpr double twoToMinus7 = 0.0078125;
constexpr double twoToMinus16 = 1.52587890625e-05;
constexpr double twoOverPi = 0.63661977236758138;
constexpr double piOver2High = 1.5707963267341256;
constexpr double piOver2Low = 6.0771005065061922e-11;

PITCH_AVX_TARGET __m128 mask4(__m256d mask) {
    int const bits = _mm256_movemask_pd(mask);
    return _mm_castsi128_ps(
        _mm_setr_epi32(-(bits & 1), -((bits >> 1) & 1), -((bits >> 2) & 1), -((bits >> 3) & 1))
    );
}
}

PITCH_AVX_TARGET void polar(
    float const* magnitude, float const* phase, float* re, float* im, size_t count
) {
    __m128 const signBit = _mm_set1_ps(-0.f), zero = _mm_setzero_ps(), one = _mm_set1_ps(1.f);
    __m128 const two = _mm_set1_ps(2.f);
    __m128 const infinity = _mm_set1_ps(std::numeric_limits<float>::infinity());
    size_t k = 0;
    for (; k + 4 <= count; k += 4) {
        __m128 const theta = _mm_loadu_ps(phase + k);
        __m128 const magnitude4 = _mm_loadu_ps(magnitude + k);
        __m128 const zeroMagnitude = _mm_cmpeq_ps(magnitude4, zero);
        if (_mm_movemask_ps(zeroMagnitude) == 0xf) {
            _mm_storeu_ps(re + k, zero);
            _mm_storeu_ps(im + k, zero);
            continue;
        }
        __m128 const absTheta = _mm_andnot_ps(signBit, theta);
        __m128 const absMagnitude = _mm_andnot_ps(signBit, magnitude4);
        __m128 const nonfinite = _mm_or_ps(
            _mm_or_ps(_mm_cmpge_ps(absTheta, infinity), _mm_cmpunord_ps(absTheta, absTheta)),
            _mm_or_ps(_mm_cmpge_ps(absMagnitude, infinity), _mm_cmpunord_ps(absMagnitude, absMagnitude))
        );
        if (_mm_movemask_ps(nonfinite)) {
            for (size_t j = k; j < k + 4; ++j) {
                if (magnitude[j] == 0) {
                    re[j] = 0.f;
                    im[j] = 0.f;
                }
                else {
                    auto const value = std::polar(magnitude[j], phase[j]);
                    re[j] = value.real();
                    im[j] = value.imag();
                }
            }
            continue;
        }

        __m256d const x = _mm256_cvtps_pd(theta);
        __m256d const ax = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
        __m256d const direct = _mm256_cmp_pd(ax, _mm256_set1_pd(piOver4), _CMP_LE_OQ);
        __m256d const small = _mm256_cmp_pd(ax, _mm256_set1_pd(twoToMinus7), _CMP_LT_OQ);
        __m256d const tiny = _mm256_cmp_pd(ax, _mm256_set1_pd(twoToMinus16), _CMP_LT_OQ);
        __m128 const direct4 = mask4(direct), small4 = mask4(small), tiny4 = mask4(tiny);

        __m256d const quotient =
            _mm256_add_pd(_mm256_mul_pd(ax, _mm256_set1_pd(twoOverPi)), _mm256_set1_pd(0.5));
        __m128i n = _mm256_cvttpd_epi32(quotient);
        n = _mm_andnot_si128(_mm_castps_si128(direct4), n);
        __m256d const fn = _mm256_cvtepi32_pd(n);
        __m256d r = _mm256_sub_pd(ax, _mm256_mul_pd(fn, _mm256_set1_pd(piOver2High)));
        r = _mm256_sub_pd(r, _mm256_mul_pd(fn, _mm256_set1_pd(piOver2Low)));
        r = _mm256_blendv_pd(r, x, direct);

        __m256d const r2 = _mm256_mul_pd(r, r);
        __m256d p = _mm256_add_pd(
            _mm256_mul_pd(_mm256_set1_pd(-2.7557319223985888e-07), r2),
            _mm256_set1_pd(2.4801587301587298e-05)
        );
        p = _mm256_add_pd(_mm256_mul_pd(p, r2), _mm256_set1_pd(-0.0013888888888888887));
        p = _mm256_add_pd(_mm256_mul_pd(p, r2), _mm256_set1_pd(0.041666666666666664));
        __m256d const r4 = _mm256_mul_pd(r2, r2);
        __m256d const cosBase =
            _mm256_sub_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(_mm256_set1_pd(0.5), r2));
        __m256d const cosP = _mm256_add_pd(_mm256_mul_pd(p, r4), cosBase);

        __m256d s = _mm256_add_pd(
            _mm256_mul_pd(_mm256_set1_pd(2.7557319223985893e-06), r2),
            _mm256_set1_pd(-0.00019841269841269841)
        );
        s = _mm256_add_pd(_mm256_mul_pd(s, r2), _mm256_set1_pd(0.0083333333333333332));
        s = _mm256_add_pd(_mm256_mul_pd(s, r2), _mm256_set1_pd(-0.16666666666666666));
        __m256d const r3 = _mm256_mul_pd(r, r2);
        __m256d const sinP = _mm256_add_pd(_mm256_mul_pd(s, r3), r);

        __m256d const x2 = _mm256_mul_pd(x, x);
        __m256d const x3 = _mm256_mul_pd(x2, x);
        __m128 const cosSmall = _mm256_cvtpd_ps(
            _mm256_sub_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(_mm256_set1_pd(0.5), x2))
        );
        __m128 const sinSmall =
            _mm256_cvtpd_ps(_mm256_sub_pd(x, _mm256_mul_pd(x3, _mm256_set1_pd(1.0 / 6.0))));

        __m128 cos = _mm256_cvtpd_ps(cosP);
        __m128 sin = _mm256_cvtpd_ps(sinP);
        cos = _mm_blendv_ps(cos, cosSmall, small4);
        sin = _mm_blendv_ps(sin, sinSmall, small4);
        cos = _mm_blendv_ps(cos, one, tiny4);
        sin = _mm_blendv_ps(sin, theta, tiny4);

        __m128 const quadrant = _mm_cvtepi32_ps(_mm_and_si128(n, _mm_set1_epi32(3)));
        __m128 const odd =
            _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_and_si128(n, _mm_set1_epi32(1)), _mm_set1_epi32(1)));
        __m128 cosValue = _mm_blendv_ps(cos, sin, odd);
        __m128 sinValue = _mm_blendv_ps(sin, cos, odd);
        __m128 const flipCos = _mm_or_ps(_mm_cmpeq_ps(quadrant, one), _mm_cmpeq_ps(quadrant, two));
        cosValue = _mm_xor_ps(cosValue, _mm_and_ps(flipCos, signBit));
        __m128 const negated = _mm_andnot_ps(direct4, _mm_and_ps(theta, signBit));
        __m128 const negative =
            _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_castps_si128(negated), _mm_castps_si128(signBit)));
        __m128 const flipSin = _mm_xor_ps(negative, _mm_cmpge_ps(quadrant, two));
        sinValue = _mm_xor_ps(sinValue, _mm_and_ps(flipSin, signBit));

        _mm_storeu_ps(re + k, _mm_andnot_ps(zeroMagnitude, _mm_mul_ps(magnitude4, cosValue)));
        _mm_storeu_ps(im + k, _mm_andnot_ps(zeroMagnitude, _mm_mul_ps(magnitude4, sinValue)));
    }
    for (; k < count; ++k) {
        if (magnitude[k] == 0) {
            re[k] = 0.f;
            im[k] = 0.f;
        }
        else {
            auto const value = std::polar(magnitude[k], phase[k]);
            re[k] = value.real();
            im[k] = value.imag();
        }
    }
    _mm256_zeroupper();
}

namespace {
constexpr double atanTable[241] = {
    0x1.ff55bb72cfde9p-5, 0x1.0f99ea71d52a6p-4, 0x1.1f86dbf082d58p-4, 0x1.2f719318a4a9ap-4,
    0x1.3f59f0e7c559dp-4, 0x1.4f3fd677292fbp-4, 0x1.5f2324fd2d7b2p-4, 0x1.6f03bdcea4b0cp-4,
    0x1.7ee182602f10ep-4, 0x1.8ebc54478fb28p-4, 0x1.9e94153cfdcf1p-4, 0x1.ae68a71c722b8p-4,
    0x1.be39ebe6f07c3p-4, 0x1.ce07c5c3cca32p-4, 0x1.ddd21701eba6ep-4, 0x1.ed98c2190043ap-4,
    0x1.fd5ba9aac2f6dp-4, 0x1.068d584212b3dp-3, 0x1.0e6adccf40881p-3, 0x1.1646541060850p-3,
    0x1.1e1fafb043726p-3, 0x1.25f6e171a535cp-3, 0x1.2dcbdb2fba1ffp-3, 0x1.359e8edeb99a3p-3,
    0x1.3d6eee8c6626cp-3, 0x1.453cec6092a9ep-3, 0x1.4d087a9da4f17p-3, 0x1.54d18ba11570ap-3,
    0x1.5c9811e3ec269p-3, 0x1.645bfffb3aa73p-3, 0x1.6c1d4898933d8p-3, 0x1.73dbde8a7d201p-3,
    0x1.7b97b4bce5b02p-3, 0x1.8350be398ebc7p-3, 0x1.8b06ee2879c28p-3, 0x1.92ba37d050271p-3,
    0x1.9a6a8e96c8626p-3, 0x1.a217e601081a5p-3, 0x1.a9c231b403279p-3, 0x1.b1696574d780bp-3,
    0x1.b90d7529260a2p-3, 0x1.c0ae54d768466p-3, 0x1.c84bf8a742e6dp-3, 0x1.cfe654e1d5395p-3,
    0x1.d77d5df205736p-3, 0x1.df110864c9d9dp-3, 0x1.e6a148e96ec4dp-3, 0x1.ee2e1451d980cp-3,
    0x1.f5b75f92c80ddp-3, 0x1.fd3d1fc40dbe4p-3, 0x1.025fa510665b5p-2, 0x1.061eea03d6290p-2,
    0x1.09dc597d86362p-2, 0x1.0d97ee509acb3p-2, 0x1.1151a362431c9p-2, 0x1.150973a9ce546p-2,
    0x1.18bf5a30bf178p-2, 0x1.1c735212dd883p-2, 0x1.2025567e47c95p-2, 0x1.23d562b381041p-2,
    0x1.278372057ef45p-2, 0x1.2b2f7fd9b5fe2p-2, 0x1.2ed987a823cfep-2, 0x1.328184fb58951p-2,
    0x1.362773707ebcbp-2, 0x1.39cb4eb76157bp-2, 0x1.3d6d129271134p-2, 0x1.410cbad6c7d32p-2,
    0x1.44aa436c2af09p-2, 0x1.4845a84d0c21bp-2, 0x1.4bdee586890e6p-2, 0x1.4f75f73869978p-2,
    0x1.530ad9951cd49p-2, 0x1.569d88e1b4cd7p-2, 0x1.5a2e0175e0f4ep-2, 0x1.5dbc3fbbe768dp-2,
    0x1.614840309cfe1p-2, 0x1.64d1ff635c1c5p-2, 0x1.685979f5fa6fdp-2, 0x1.6bdeac9cbd76cp-2,
    0x1.6f61941e4def0p-2, 0x1.72e22d53aa2a9p-2, 0x1.7660752817501p-2, 0x1.79dc6899118d1p-2,
    0x1.7d5604b63b3f7p-2, 0x1.80cd46a14b1d0p-2, 0x1.84422b8df95d7p-2, 0x1.87b4b0c1ebedbp-2,
    0x1.8b24d394a1b25p-2, 0x1.8e92916f5cde8p-2, 0x1.91fde7cd0c662p-2, 0x1.9566d43a34907p-2,
    0x1.98cd5454d6b18p-2, 0x1.9c3165cc58107p-2, 0x1.9f93066168001p-2, 0x1.a2f233e5e530bp-2,
    0x1.a64eec3cc23fcp-2, 0x1.a9a92d59e98cfp-2, 0x1.ad00f5422058bp-2, 0x1.b056420ae9343p-2,
    0x1.b3a911da65c6cp-2, 0x1.b6f962e737efbp-2, 0x1.ba473378624a5p-2, 0x1.bd9281e528191p-2,
    0x1.c0db4c94ec9efp-2, 0x1.c42191ff11eb6p-2, 0x1.c76550aad71f8p-2, 0x1.caa6872f3631bp-2,
    0x1.cde53432c1350p-2, 0x1.d121566b7f2adp-2, 0x1.d45aec9ec862bp-2, 0x1.d791f5a1226f4p-2,
    0x1.dac670561bb4fp-2, 0x1.ddf85bb026974p-2, 0x1.e127b6b0744afp-2, 0x1.e4548066cf51ap-2,
    0x1.e77eb7f175a34p-2, 0x1.eaa65c7cf28c4p-2, 0x1.edcb6d43f8434p-2, 0x1.f0ede98f393cfp-2,
    0x1.f40dd0b541417p-2, 0x1.f72b221a4e495p-2, 0x1.fa45dd3029258p-2, 0x1.fd5e0175fdf83p-2,
    0x1.0039c73c1a40bp-1, 0x1.01c341e82422dp-1, 0x1.034b709250488p-1, 0x1.04d25314342e5p-1,
    0x1.0657e94db30cfp-1, 0x1.07dc3324e9b38p-1, 0x1.095f30861a58fp-1, 0x1.0ae0e1639866cp-1,
    0x1.0c6145b5b43dap-1, 0x1.0de05d7aa6f7cp-1, 0x1.0f5e28b67e295p-1, 0x1.10daa77307a0dp-1,
    0x1.1255d9bfbd2a8p-1, 0x1.13cfbfb1b056ep-1, 0x1.1548596376469p-1, 0x1.16bfa6f5137e1p-1,
    0x1.1835a88be7c13p-1, 0x1.19aa5e5299f99p-1, 0x1.1b1dc87904284p-1, 0x1.1c8fe7341f64fp-1,
    0x1.1e00babdefeb3p-1, 0x1.1f7043557138ap-1, 0x1.20de813e823b1p-1, 0x1.224b74c1d192ap-1,
    0x1.23b71e2cc9e6ap-1, 0x1.25217dd17e501p-1, 0x1.268a940696da6p-1, 0x1.27f261273d1b3p-1,
    0x1.2958e59308e30p-1, 0x1.2abe21aded073p-1, 0x1.2c2215e024465p-1, 0x1.2d84c2961e48bp-1,
    0x1.2ee628406cbcap-1, 0x1.30464753b090ap-1, 0x1.31a52048874bep-1, 0x1.3302b39b78856p-1,
    0x1.345f01cce37bbp-1, 0x1.35ba0b60ecccep-1, 0x1.3713d0df6c503p-1, 0x1.386c52d3db11ep-1,
    0x1.39c391cd41719p-1, 0x1.3b198e5e2564ap-1, 0x1.3c6e491c78dc4p-1, 0x1.3dc1c2a188504p-1,
    0x1.3f13fb89e96f4p-1, 0x1.4064f47569f48p-1, 0x1.41b4ae06fea41p-1, 0x1.430328e4b26d5p-1,
    0x1.445065b795b55p-1, 0x1.459c652badc7fp-1, 0x1.46e727efe4715p-1, 0x1.4830aeb5f7bfdp-1,
    0x1.4978fa3269ee1p-1, 0x1.4ac00b1c71762p-1, 0x1.4c05e22de94e4p-1, 0x1.4d4a8023414e8p-1,
    0x1.4e8de5bb6ec04p-1, 0x1.4fd013b7dd17ep-1, 0x1.51110adc5ed81p-1, 0x1.5250cbef1e9fap-1,
    0x1.538f57b89061ep-1, 0x1.54ccaf0362c8fp-1, 0x1.5608d29c70c34p-1, 0x1.5743c352b33b9p-1,
    0x1.587d81f732fbap-1, 0x1.59b60f5cfab9dp-1, 0x1.5aed6c5909517p-1, 0x1.5c2399c244260p-1,
    0x1.5d58987169b18p-1, 0x1.5e8c6941043cfp-1, 0x1.5fbf0d0d5cc49p-1, 0x1.60f084b46e05ep-1,
    0x1.6220d115d7b8dp-1, 0x1.634ff312d1f3bp-1, 0x1.647deb8e20b8fp-1, 0x1.65aabb6c07b02p-1,
    0x1.66d663923e086p-1, 0x1.6800e4e7e2857p-1, 0x1.692a40556fb6ap-1, 0x1.6a5276c4b0575p-1,
    0x1.6b798920b3d98p-1, 0x1.6c9f7855c3198p-1, 0x1.6dc44551553aep-1, 0x1.6ee7f10204aefp-1,
    0x1.700a7c5784633p-1, 0x1.712be84295198p-1, 0x1.724c35b4fae7bp-1, 0x1.736b65a172dffp-1,
    0x1.748978fba8e0fp-1, 0x1.75a670b82d8d8p-1, 0x1.76c24dcc6c6c0p-1, 0x1.77dd112ea22c7p-1,
    0x1.78f6bbd5d315ep-1, 0x1.7a0f4eb9c19a2p-1, 0x1.7b26cad2e50fdp-1, 0x1.7c3d311a6092bp-1,
    0x1.7d528289fa093p-1, 0x1.7e66c01c114fdp-1, 0x1.7f79eacb97898p-1, 0x1.808c03940694ap-1,
    0x1.819d0b7158a4cp-1, 0x1.82ad036000005p-1, 0x1.83bbec5cdee22p-1, 0x1.84c9c7653f7eap-1,
    0x1.85d69576cc2c5p-1, 0x1.86e2578f87ae5p-1, 0x1.87ed0eadc5a2ap-1, 0x1.88f6bbd023118p-1,
    0x1.89ff5ff57f1f7p-1, 0x1.8b06fc1cf3dfep-1, 0x1.8c0d9145cf49dp-1, 0x1.8d13206f8c4cap-1,
    0x1.8e17aa99cc05dp-1, 0x1.8f1b30c44f167p-1, 0x1.901db3eeef187p-1, 0x1.911f35199833bp-1,
    0x1.921fb54442d18p-1,
};

constexpr double pi = 3.141592653589793;
constexpr double piOver2 = 1.5707963267948966;

PITCH_AVX_TARGET __m256d signLanes(int mask) {
    return _mm256_castsi256_pd(_mm256_set_epi64x(
        (mask & 8) ? -1 : 0, (mask & 4) ? -1 : 0, (mask & 2) ? -1 : 0, (mask & 1) ? -1 : 0
    ));
}
}

PITCH_AVX_TARGET void arguments(float const* re, float const* im, float* out, size_t count) {
    size_t k = 0;
    for (; k + 4 <= count; k += 4) {
        __m256d const signBit =
            _mm256_castsi256_pd(_mm256_set1_epi64x(static_cast<long long>(0x8000000000000000ull)));
        __m256d const y = _mm256_cvtps_pd(_mm_loadu_ps(im + k));
        __m256d const x = _mm256_cvtps_pd(_mm_loadu_ps(re + k));
        __m256d const ay = _mm256_andnot_pd(signBit, y);
        __m256d const ax = _mm256_andnot_pd(signBit, x);
        __m256d const zero = _mm256_setzero_pd(), one = _mm256_set1_pd(1.0);
        __m256d const xZero = _mm256_cmp_pd(ax, zero, _CMP_EQ_OQ);
        __m256d const bothZero = _mm256_and_pd(xZero, _mm256_cmp_pd(ay, zero, _CMP_EQ_OQ));
        // A both-zero lane divides by 1, since dividing by zero raises a floating-point exception.
        __m256d const bigger = _mm256_blendv_pd(_mm256_max_pd(ax, ay), one, bothZero);
        __m256d const smaller = _mm256_min_pd(ax, ay);
        __m256d const swap = _mm256_cmp_pd(ay, ax, _CMP_GT_OQ);
        __m256d const z = _mm256_div_pd(smaller, bigger);

        __m256d const z2 = _mm256_mul_pd(z, z);
        __m256d p = _mm256_set1_pd(0.1428571356180717);
        p = _mm256_sub_pd(_mm256_set1_pd(0.19999999999393223), _mm256_mul_pd(p, z2));
        p = _mm256_sub_pd(_mm256_set1_pd(0.3333333333333317), _mm256_mul_pd(p, z2));
        __m256d const smallPoly = _mm256_sub_pd(z, _mm256_mul_pd(_mm256_mul_pd(z2, z), p));
        __m256d const tinyAngle =
            _mm256_blendv_pd(smallPoly, z, _mm256_cmp_pd(z, _mm256_set1_pd(0.0001), _CMP_LT_OQ));

        __m256d const zc = _mm256_max_pd(z, _mm256_set1_pd(0.0625));
        __m128i const index = _mm256_cvttpd_epi32(
            _mm256_add_pd(_mm256_mul_pd(zc, _mm256_set1_pd(256.0)), _mm256_set1_pd(0.5))
        );
        __m256d const idx = _mm256_cvtepi32_pd(index);
        __m256d const table = _mm256_set_pd(
            atanTable[_mm_extract_epi32(index, 3) - 16],
            atanTable[_mm_extract_epi32(index, 2) - 16],
            atanTable[_mm_extract_epi32(index, 1) - 16],
            atanTable[_mm_extract_epi32(index, 0) - 16]
        );
        __m256d const num =
            _mm256_sub_pd(_mm256_mul_pd(smaller, _mm256_set1_pd(256.0)), _mm256_mul_pd(idx, bigger));
        __m256d const den =
            _mm256_add_pd(_mm256_mul_pd(bigger, _mm256_set1_pd(256.0)), _mm256_mul_pd(smaller, idx));
        __m256d const t = _mm256_div_pd(num, den);
        __m256d const cubic =
            _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(t, t), t), _mm256_set1_pd(0.33333333333224097));
        __m256d const tableAngle = _mm256_sub_pd(_mm256_add_pd(t, table), cubic);
        __m256d angle = _mm256_blendv_pd(
            tableAngle, tinyAngle, _mm256_cmp_pd(z, _mm256_set1_pd(0.0625), _CMP_LE_OQ)
        );

        __m256d const xNeg = signLanes(_mm256_movemask_pd(x));
        __m256d const yNeg = signLanes(_mm256_movemask_pd(y));
        angle = _mm256_blendv_pd(angle, _mm256_sub_pd(_mm256_set1_pd(piOver2), angle), swap);
        angle = _mm256_blendv_pd(angle, _mm256_sub_pd(_mm256_set1_pd(pi), angle), xNeg);
        angle = _mm256_blendv_pd(angle, _mm256_xor_pd(angle, signBit), yNeg);

        __m256i const exponentMask = _mm256_set1_epi64x(0x7ff);
        __m256i const yBits = _mm256_castpd_si256(y), xBits = _mm256_castpd_si256(x);
        __m128i const yLo = _mm256_castsi256_si128(yBits), xLo = _mm256_castsi256_si128(xBits);
        __m128i const yHi = _mm256_extractf128_si256(yBits, 1);
        __m128i const xHi = _mm256_extractf128_si256(xBits, 1);
        __m128i const exp = _mm256_castsi256_si128(exponentMask);
        __m128i const dLo = _mm_sub_epi64(
            _mm_and_si128(_mm_srli_epi64(yLo, 52), exp), _mm_and_si128(_mm_srli_epi64(xLo, 52), exp)
        );
        __m128i const dHi = _mm_sub_epi64(
            _mm_and_si128(_mm_srli_epi64(yHi, 52), exp), _mm_and_si128(_mm_srli_epi64(xHi, 52), exp)
        );
        __m256d const tinyMask = _mm256_castsi256_pd(_mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_cmpgt_epi64(_mm_set1_epi64x(-13), dLo)),
            _mm_cmpgt_epi64(_mm_set1_epi64x(-13), dHi),
            1
        ));
        angle = _mm256_blendv_pd(
            angle, _mm256_div_pd(y, _mm256_blendv_pd(x, one, xZero)), _mm256_andnot_pd(xNeg, tinyMask)
        );

        __m256d const signedPi = _mm256_or_pd(_mm256_set1_pd(pi), _mm256_and_pd(y, signBit));
        angle = _mm256_blendv_pd(angle, _mm256_blendv_pd(y, signedPi, xNeg), bothZero);

        _mm_storeu_ps(out + k, _mm256_cvtpd_ps(angle));

        __m256d const inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());
        __m256d const nonfinite = _mm256_or_pd(
            _mm256_or_pd(_mm256_cmp_pd(ay, inf, _CMP_EQ_OQ), _mm256_cmp_pd(ax, inf, _CMP_EQ_OQ)),
            _mm256_or_pd(_mm256_cmp_pd(y, y, _CMP_UNORD_Q), _mm256_cmp_pd(x, x, _CMP_UNORD_Q))
        );
        int bad = _mm256_movemask_pd(nonfinite);
        while (bad) {
    #ifdef _MSC_VER
            unsigned long lane = 0;
            _BitScanForward(&lane, static_cast<unsigned long>(bad));
    #else
            int const lane = __builtin_ctz(static_cast<unsigned>(bad));
    #endif
            out[k + lane] = std::arg(std::complex<float>(re[k + lane], im[k + lane]));
            bad &= bad - 1;
        }
    }
    for (; k < count; ++k)
        out[k] = std::arg(std::complex<float>(re[k], im[k]));
    _mm256_zeroupper();
}
} // namespace pitch_simd
#else
namespace pitch_simd {
bool available() {
    return false;
}

bool exactFloatMath() {
    return false;
}

// available() is false off x86-64, so this is never called; stop loudly if that ever changes.
void butterflies(float*, float*, float*, float*, size_t, float const*, float const*) {
    std::abort();
}

void magnitudes(float const*, float const*, float*, size_t) {
    std::abort();
}

void arguments(float const*, float const*, float*, size_t) {
    std::abort();
}

void polar(float const*, float const*, float*, float*, size_t) {
    std::abort();
}
} // namespace pitch_simd
#endif
