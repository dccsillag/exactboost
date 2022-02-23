#pragma once

#include <cassert>
#include <algorithm>
#include <vector>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#ifndef NO_PYTHON
#  include <pybind11/pybind11.h>
#endif


// We use these to avoid overflow.
using metric_t = long long int;
using length_t = unsigned int;

struct PosNeg {
    PosNeg() = default;
    PosNeg(length_t n_pos, length_t n_neg);
    PosNeg(const std::vector<float>& values, const std::vector<uint8_t>& labels);

    std::vector<float> pos;
    std::vector<float> neg;
};

struct Interval {
    Interval() = default;
    Interval(float inf, float sup);
    Interval(float x);

    [[nodiscard]] float mid() const;

    [[nodiscard]] Interval fst() const;
    [[nodiscard]] Interval snd() const;

    friend bool operator==(const Interval& lhs, const Interval& rhs);

    friend std::ostream& operator <<(std::ostream& os, const Interval& i);

    float inf;
    float sup;
};

struct StumpGivenFeature {
    StumpGivenFeature() = default;
    StumpGivenFeature(float xi, float a, float b);

    float operator()(float x) const;
    friend std::ostream& operator <<(std::ostream& os, const StumpGivenFeature& s);

    void apply(const std::vector<float>& feature,
               const std::vector<float>& score_in,
               std::vector<float>& score_out) const;

    float xi;
    float a;
    float b;
};

struct IntervalStumpGivenFeature {
    IntervalStumpGivenFeature() = default;
    IntervalStumpGivenFeature(const Interval& xi, const Interval& a, const Interval& b);

    friend std::ostream& operator <<(std::ostream& os, const IntervalStumpGivenFeature& s);

    std::pair<std::vector<float>::const_iterator, std::vector<float>::const_iterator>
        static prepare(const std::vector<float>& feature,
                       const std::vector<float>& score_in,
                       std::vector<float>& score_out,
                       Interval xi);
    void apply(const std::vector<float>& feature,
               const std::vector<float>& score_in,
               const std::pair<std::vector<float>::const_iterator, std::vector<float>::const_iterator>& sep,
               std::vector<float>& score_out,
               bool is_positive) const;

    Interval xi;
    Interval a;
    Interval b;
};

struct Stump {
    Stump() = default;
    Stump(length_t feature, StumpGivenFeature base_stump);
    Stump(length_t feature, float xi, float a, float b);

    void add(const std::vector<std::vector<float>>& X,
             const std::vector<float>& feature_scales,
             std::vector<float>& score) const;

    friend std::ostream& operator<<(std::ostream& os, const Stump& s);

    length_t feature;
    StumpGivenFeature base_stump;
};

void normalize01(std::vector<float>& vec);

template <typename T>
void sort_together(std::vector<T>& vec, std::vector<T>& aux) {
    /* sorts by `vec`. */
    assert(vec.size() == aux.size());
    const length_t n = vec.size();

    std::vector<std::pair<T, T>> data(n);
    for (length_t i = 0; i < n; ++i)
        data[i] = {vec[i], aux[i]};

    std::sort(data.begin(), data.end(), [](const auto& l, const auto& r) {
        return l.first < r.first;
    });

    std::transform(data.cbegin(), data.cend(), vec.begin(), [](const auto& p) { return p. first; });
    std::transform(data.cbegin(), data.cend(), aux.begin(), [](const auto& p) { return p.second; });
}

template <typename T>
void apply_indices(std::vector<T>& vec, std::vector<length_t>& indices) {
    std::vector<T> out(indices.size());

    for (length_t i = 0; i < indices.size(); ++i)
        out[i] = vec[indices[i]];

    vec.swap(out);
}

template <typename T>
std::vector<T> xvector_to_vec(const xt::xtensor<T, 1>& vec) {
    std::vector<T> out(vec.shape(0));
    std::copy(vec.cbegin(), vec.cend(), out.begin());
    return out;
}

template <typename T>
std::vector<std::vector<T>> xmatrix_to_vecvec(const xt::xtensor<T, 2>& mat) {
    const size_t p = mat.shape(1);

    std::vector<std::vector<T>> out(p, std::vector<T>(mat.shape(0)));
    for(length_t j = 0; j < p; ++j) {
        xt::xview feature = xt::col(mat, j);
        std::copy(feature.cbegin(), feature.cend(), out[j].begin());
    }

    return out;
}

bool detect_interrupt(bool& should_break);
