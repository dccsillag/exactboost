#include "utils.hpp"

#include <algorithm>


PosNeg::PosNeg(length_t n_pos, length_t n_neg)
    : pos(n_pos)
    , neg(n_neg)
{ }

PosNeg::PosNeg(const std::vector<float>& values, const std::vector<uint8_t>& labels)
    : pos()
    , neg()
{
    assert(values.size() == labels.size());
    const length_t n = values.size();

    for (length_t i = 0; i < n; ++i) {
        if ((bool)labels[i])
            pos.push_back(values[i]);
        else
            neg.push_back(values[i]);
    }
}

Interval::Interval(float inf, float sup)
    : inf(inf)
    , sup(sup)
{ }

Interval::Interval(float x)
    : inf(x)
    , sup(x)
{ }

float Interval::mid() const {
    return (inf + sup)*0.5F;
}

Interval Interval::fst() const {
    return Interval(inf, mid());
}

Interval Interval::snd() const {
    return Interval(mid(), sup);
}

bool operator==(const Interval& lhs, const Interval& rhs) {
    return lhs.inf == rhs.inf && lhs.sup == rhs.sup;
}

std::ostream& operator <<(std::ostream& os, const Interval& i) {
    os << "[" << i.inf << ", " << i.sup << "]";
    return os;
}

StumpGivenFeature::StumpGivenFeature(float xi, float a, float b)
    : xi(xi)
    , a(a)
    , b(b)
{ }

float StumpGivenFeature::operator()(float x) const {
    return x <= xi ? a : b;
}

std::ostream& operator <<(std::ostream& os, const StumpGivenFeature& s) {
    os << "StumpGivenFeature(xi=" << s.xi << ", a=" << s.a << ", b=" << s.b << ")";
    return os;
}

void StumpGivenFeature::apply(const std::vector<float>& feature,
                              const std::vector<float>& score_in,
                              std::vector<float>& score_out) const {
    assert(std::is_sorted(score_in.cbegin(), score_in.cend()));

    // partition
    std::vector<float>::iterator sep;
    {
        auto out = score_out.begin();
        for (auto ifeature = feature.begin(), iscore_in = score_in.begin();
             ifeature < feature.end() && iscore_in < score_in.end();
             ++ifeature, ++iscore_in)
            if (*ifeature <= xi) *(out++) = *iscore_in;
        sep = out;
        for (auto ifeature = feature.begin(), iscore_in = score_in.begin();
             ifeature < feature.end() && iscore_in < score_in.end();
             ++ifeature, ++iscore_in)
            if (*ifeature > xi) *(out++) = *iscore_in;
    }

    // add values
    for (auto i = score_out.begin(); i < sep; ++i) *i += a;
    for (auto i = sep; i < score_out.end();   ++i) *i += b;

    // merge
    std::inplace_merge(score_out.begin(), sep, score_out.end());

    assert(std::is_sorted(score_in.cbegin(), score_in.cend()));
}

IntervalStumpGivenFeature::IntervalStumpGivenFeature(const Interval& xi, const Interval& a, const Interval& b)
    : xi(xi)
    , a(a)
    , b(b)
{ }

std::ostream& operator <<(std::ostream& os, const IntervalStumpGivenFeature& s) {
    os << "StumpGivenFeature(xi=" << s.xi << ", a=" << s.a << ", b=" << s.b << ")";
    return os;
}

std::pair<std::vector<float>::const_iterator, std::vector<float>::const_iterator>
    IntervalStumpGivenFeature::prepare(const std::vector<float>& feature,
                                       const std::vector<float>& score_in,
                                       std::vector<float>& score_out,
                                       Interval xi) {
    assert(std::is_sorted(score_in.cbegin(), score_in.cend()));

    // partition
    std::vector<float>::iterator sep0;
    std::vector<float>::iterator sep1;

    auto out = score_out.begin();
    for (auto ifeature = feature.begin(), iscore_in = score_in.begin();
         ifeature < feature.end() && iscore_in < score_in.end();
         ++ifeature, ++iscore_in)
        if (*ifeature <= xi.inf) *(out++) = *iscore_in;
    sep0 = out;
    for (auto ifeature = feature.begin(), iscore_in = score_in.begin();
         ifeature < feature.end() && iscore_in < score_in.end();
         ++ifeature, ++iscore_in)
        if (*ifeature > xi.inf && *ifeature <= xi.sup) *(out++) = *iscore_in;
    sep1 = out;
    for (auto ifeature = feature.begin(), iscore_in = score_in.begin();
         ifeature < feature.end() && iscore_in < score_in.end();
         ++ifeature, ++iscore_in)
        if (*ifeature > xi.sup) *(out++) = *iscore_in;

    return {sep0, sep1};
}

void IntervalStumpGivenFeature::apply(const std::vector<float>& feature,
                                      const std::vector<float>& score_in,
                                      const std::pair<std::vector<float>::const_iterator,
                                                      std::vector<float>::const_iterator>& sep,
                                      std::vector<float>& score_out,
                                      bool is_positive) const {
    assert(std::is_sorted(score_in.cbegin(), sep.first));
    assert(std::is_sorted(sep.first, sep.second));
    assert(std::is_sorted(sep.second, score_in.cend()));

    // get a and b, as well as their min/max with 0
    float _a, _b, _a0, _b0;
    if (is_positive) {
        _a = a.sup, _b = b.sup;
        _a0 = std::max(0.0F, _a), _b0 = std::max(0.0F, _b);
    } else {
        _a = a.inf, _b = b.inf;
        _a0 = std::min(0.0F, _a), _b0 = std::min(0.0F, _b);
    }

    std::copy(score_in.cbegin(), score_in.cend(), score_out.begin());
    auto sep0 = score_out.begin() + (sep.first - score_in.cbegin());
    auto sep1 = score_out.begin() + (sep.second - score_in.cbegin());
    for (auto i = score_out.begin(); i <            sep0; ++i) *i += _a;
    for (auto i =              sep0; i <            sep1; ++i) *i += _a0 + _b0;
    for (auto i =              sep1; i < score_out.end(); ++i) *i += _b;

    // merge
    std::inplace_merge(score_out.begin(), sep0, sep1);
    std::inplace_merge(score_out.begin(), sep1, score_out.end());

    assert(std::is_sorted(score_out.cbegin(), score_out.cend()));
}

Stump::Stump(length_t feature, StumpGivenFeature base_stump)
    : feature(feature)
    , base_stump(base_stump)
{ }

Stump::Stump(length_t feature, float xi, float a, float b)
    : Stump(feature, {xi, a, b})
{ }

void Stump::add(const std::vector<std::vector<float>>& X,
                const std::vector<float>& feature_scales,
                std::vector<float>& score) const {
    const length_t n = score.size();

    float feature_scale = feature_scales[feature];
    const std::vector<float>& feature_vec = X[feature];

    for (length_t i = 0; i < n; ++i)
        score[i] += base_stump(feature_vec[i] * feature_scale);
}

std::ostream& operator<<(std::ostream& os, const Stump& s) {
    os << "Stump(j=" << s.feature << ", xi=" << s.base_stump.xi << ", a=" << s.base_stump.a << ", b=" << s.base_stump.b << ")";
    return os;
}

void normalize01(std::vector<float>& vec) {
    auto [emin, emax] = std::minmax_element(vec.cbegin(), vec.cend());
    float min = *emin;
    float max = *emax;

    if (max == min)
        for (float& x : vec) x = 0.0F;

    for (float& x : vec) {
        x -= min;
        x *= 1.0F/(max - min);
    }
}

bool detect_interrupt(bool& should_break) {
#ifndef NO_PYTHON
    bool ret = false;
#pragma omp critical
    {
        should_break |= PyErr_CheckSignals() != 0;
        ret = should_break;
    }
    return ret;
#endif
}
