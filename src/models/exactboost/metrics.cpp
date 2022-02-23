#include "metrics.hpp"

#include <cassert>
#include <algorithm>


AUC::AUC() = default;
KS::KS() = default;
PaK::PaK(length_t k) : k(k) { }


metric_t AUC::operator()(const PosNeg& score) const {
    assert(std::is_sorted(score.pos.cbegin(), score.pos.cend()));
    assert(std::is_sorted(score.neg.cbegin(), score.neg.cend()));

    metric_t auc = 0;
    metric_t n_neg = 0;

    auto i = score.pos.cbegin();
    auto j = score.neg.cbegin();
    while (i < score.pos.cend() && j < score.neg.cend()) {
        if (*i < *j) { // label = 1
            auc += n_neg;
            ++i;
        } else { // label = 0
            ++n_neg;
            ++j;
        }
    }
    auc += n_neg*std::distance(i, score.pos.cend());

    return auc;
}

std::unique_ptr<BaseMetric> AUC::with_subsampling(double observation_subsampling) const {
    return std::unique_ptr<BaseMetric>(new AUC());
}

metric_t KS::operator()(const PosNeg& score) const {
    assert(std::is_sorted(score.pos.cbegin(), score.pos.cend()));
    assert(std::is_sorted(score.neg.cbegin(), score.neg.cend()));

    metric_t n1 = (metric_t)score.pos.size();
    metric_t n0 = (metric_t)score.neg.size();

    metric_t best_ks = std::numeric_limits<metric_t>::min();
    metric_t passing_ks = 0;

    auto i = score.pos.cbegin();
    auto j = score.neg.cbegin();
    while (i < score.pos.cend() && j < score.neg.cend()) {
        if (*i < *j) { // label = 1
            passing_ks -= n0;
            ++i;
        } else { // label = 0
            passing_ks += n1;
            ++j;
        }
        best_ks = std::max(best_ks, passing_ks);
    }
    passing_ks -= n0*std::distance(i, score.pos.cend());
    passing_ks += n1*std::distance(j, score.neg.cend());
    best_ks = std::max(best_ks, passing_ks);

    return best_ks;
}

std::unique_ptr<BaseMetric> KS::with_subsampling(double observation_subsampling) const {
    return std::unique_ptr<BaseMetric>(new KS());
}

metric_t PaK::operator()(const PosNeg& score) const {
    assert(std::is_sorted(score.pos.cbegin(), score.pos.cend()));
    assert(std::is_sorted(score.neg.cbegin(), score.neg.cend()));

    metric_t pak = 0;
    auto i = score.pos.crbegin();
    auto j = score.neg.crbegin();
    length_t _k = k;
    while (i < score.pos.crend() && j < score.neg.crend() && _k > 0) {
        if (*i > *j) { // label = 1
            ++pak;
            ++i;
        } else { // label = 0
            ++j;
        }
        --_k;
    }
    pak += (metric_t)std::min(_k, (length_t)std::distance(i, score.pos.crend()));

    // FIXME: this is in [0, k], but in the progress bar code we assume this is in [0, n0*n1].

    return pak;
}

std::unique_ptr<BaseMetric> PaK::with_subsampling(double observation_subsampling) const {
    return std::unique_ptr<BaseMetric>(new PaK(ceil((double)k * observation_subsampling)));
}
