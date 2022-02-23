#include "exactboost.hpp"

#include <algorithm>
#include <numeric>
#include <limits>
#include <xtensor/xrandom.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xadapt.hpp>
#include <fmt/core.h>


ExactBoost::ExactBoost(const BaseMetric& metric, float margin,
                       unsigned int n_estimators, unsigned int n_rounds,
                       double observation_subsampling, double feature_subsampling)
    : metric(metric)
    , margin(margin)
    , n_estimators(n_estimators)
    , n_rounds(n_rounds)
    , observation_subsampling(observation_subsampling)
    , feature_subsampling(feature_subsampling)
{
    for (unsigned int k = 0; k < n_estimators; ++k)
        learned_estimators.push_back(ExactBoostSingle(metric, margin,
                                                      n_rounds,
                                                      observation_subsampling, feature_subsampling,
                                                      std::mt19937{k}));
}

ExactBoost::ExactBoost(const BaseMetric& metric, float margin,
                       unsigned int n_estimators, unsigned int n_rounds,
                       double observation_subsampling, double feature_subsampling,
                       const std::vector<float>& feature_scales,
                       const std::vector<std::vector<Stump>>& learned_stumps)
    : metric(metric)
    , margin(margin)
    , n_estimators(n_estimators)
    , n_rounds(n_rounds)
    , observation_subsampling(observation_subsampling)
    , feature_subsampling(feature_subsampling)
    , feature_scales(feature_scales)
{
    for (unsigned int k = 0; k < n_estimators; ++k)
        learned_estimators.push_back(ExactBoostSingle(metric, margin,
                                                      n_rounds,
                                                      observation_subsampling, feature_subsampling,
                                                      learned_stumps[k],
                                                      std::mt19937{k}));
}

ExactBoostSingle::ExactBoostSingle(const BaseMetric& metric, float margin,
                                   unsigned int n_rounds,
                                   double observation_subsampling, double feature_subsampling,
                                   std::mt19937&& rng)
    : metric(metric)
    , margin(margin)
    , n_rounds(n_rounds)
    , observation_subsampling(observation_subsampling)
    , feature_subsampling(feature_subsampling)
    , rng(rng)
{ }

ExactBoostSingle::ExactBoostSingle(const BaseMetric& metric, float margin,
                                   unsigned int n_rounds,
                                   double observation_subsampling, double feature_subsampling,
                                   std::vector<Stump> learned_stumps,
                                   std::mt19937&& rng)
    : learned_stumps(std::move(learned_stumps))
    , metric(metric)
    , margin(margin)
    , n_rounds(n_rounds)
    , observation_subsampling(observation_subsampling)
    , feature_subsampling(feature_subsampling)
    , rng(rng)
{ }

void ExactBoost::fit(const std::vector<std::vector<float>>& X,
                     const std::vector<uint8_t>& y,
                     bool interaction) {
    const length_t p = X.size();
    feature_scales.insert(feature_scales.end(), p, 0);
    for (length_t j = 0; j < p; ++j) {
        const std::vector<float>& feature = X[j];
        auto [min_feature_value_iter, max_feature_value_iter] =
            std::minmax_element(feature.cbegin(), feature.cend());
        float min_feature_value = *min_feature_value_iter;
        float max_feature_value = *max_feature_value_iter;
        feature_scales[j] = 1.0F / std::max(std::abs(max_feature_value), std::abs(min_feature_value));
    }

    ProgressBar pg(n_estimators, n_rounds, metric, y);

    bool should_break = false;
#pragma omp parallel for default(none) shared(X, y, interaction, pg, should_break)
    for (unsigned int k = 0; k < n_estimators; ++k) {
        if (should_break)
            continue;

        learned_estimators[k].fit(X, y, interaction,
                                  pg.get_feed_func(k), feature_scales, should_break);

        if (interaction) detect_interrupt(should_break);
    }

#ifndef NO_PYTHON
    if (should_break)
        throw pybind11::error_already_set();
#endif

    pg.finish();
}

std::vector<float> ExactBoost::predict(const std::vector<std::vector<float>>& X) {
    const length_t n = X.front().size();

    std::vector<float> out(n, 0);
    for (auto& estimator : learned_estimators) {
        std::vector<float> pred = estimator.predict(X, feature_scales);
        for (length_t i = 0; i < n; ++i)
            out[i] += pred[i];
    }

    float inv_n_estimators = 1.0F / (float)n_estimators;
    for (length_t i = 0; i < n; ++i)
        out[i] *= inv_n_estimators;

    normalize01(out);

    return out;
}

void ExactBoost::fit(const xt::xtensor<float, 2>& X, const xt::xtensor<uint8_t, 1>& y,
                     bool interaction) {
    fit(xmatrix_to_vecvec(X), xvector_to_vec(y), interaction);
}

xt::xtensor<float, 1> ExactBoost::predict(const xt::xtensor<float, 2>& X) {
    std::vector<float> preds = predict(xmatrix_to_vecvec(X));
    return xt::adapt(preds, {X.shape(0)});
}

void ExactBoostSingle::fit(const std::vector<std::vector<float>>& X,
                           const std::vector<uint8_t>& y,
                           bool interaction,
                           const ProgressBar::feed_func& feed_pg,
                           const std::vector<float>& feature_scales,
                           bool& should_break) {
    const length_t p = X.size();
    const length_t n = y.size();
    const length_t n1 = std::count(y.cbegin(), y.cend(), 1);
    const length_t n0 = std::count(y.cbegin(), y.cend(), 0);
    const length_t p_sub = std::ceil((double)p * feature_subsampling);
    const length_t n1_sub = std::ceil((double)n1 * observation_subsampling);
    const length_t n0_sub = std::ceil((double)n0 * observation_subsampling);

    std::vector<float> score = generate_random_score(n);
    metric_t best_metric = std::numeric_limits<metric_t>::min();

    PosNeg workspace1(n1_sub, n0_sub);
    PosNeg workspace2(n1_sub, n0_sub);

    std::unique_ptr<BaseMetric> metric_subsampled = metric.with_subsampling(observation_subsampling);

    for (unsigned int t = 0; t < n_rounds; ++t) {
        if (interaction && detect_interrupt(should_break)) return;

        // Subsample features
        std::vector<length_t> features_to_consider(p_sub);
        std::vector<length_t> pos_observations_to_consider(n1_sub);
        std::vector<length_t> neg_observations_to_consider(n0_sub);
        {
            std::vector<length_t> indices(p);
            std::iota(indices.begin(), indices.end(), 0);
            if (feature_subsampling < 1.0)
                std::sample(indices.cbegin(), indices.cend(), features_to_consider.begin(), p_sub, rng);
            else
                std::copy(indices.cbegin(), indices.cend(), features_to_consider.begin());
        }
        {
            std::uniform_int_distribution<length_t> dist(0, n1-1);
            if (observation_subsampling < 1.0)
                std::generate(pos_observations_to_consider.begin(), pos_observations_to_consider.end(),
                              [this, &dist]() { return dist(rng); });
            else
                std::iota(pos_observations_to_consider.begin(), pos_observations_to_consider.end(), 0);
        }
        {
            std::uniform_int_distribution<length_t> dist(0, n0-1);
            if (observation_subsampling < 1.0)
                std::generate(neg_observations_to_consider.begin(), neg_observations_to_consider.end(),
                              [this, &dist]() { return dist(rng); });
            else
                std::iota(neg_observations_to_consider.begin(), neg_observations_to_consider.end(), 0);
        }

        Stump proposed_stump;
        {
            Stump new_best_stump;
            metric_t new_best_metric = best_metric;
            PosNeg score_posneg(score, y);
            apply_indices(score_posneg.pos, pos_observations_to_consider);
            apply_indices(score_posneg.neg, neg_observations_to_consider);
            for (length_t j : features_to_consider) {
                const std::vector<float>& feature = X[j];
                auto [min_feature_value_iter, max_feature_value_iter] =
                    std::minmax_element(feature.cbegin(), feature.cend());
                float min_feature_value = *min_feature_value_iter;
                float max_feature_value = *max_feature_value_iter;
                float scale = feature_scales[j];

                PosNeg feature_posneg(feature, y);
                apply_indices(feature_posneg.pos, pos_observations_to_consider);
                apply_indices(feature_posneg.neg, neg_observations_to_consider);
                if (scale != INFINITY) {
                    for (float& x : feature_posneg.pos) x *= scale;
                    for (float& x : feature_posneg.neg) x *= scale;
                    min_feature_value *= scale;
                    max_feature_value *= scale;
                }

                sort_together(score_posneg.pos, feature_posneg.pos);
                sort_together(score_posneg.neg, feature_posneg.neg);

                if (interaction && detect_interrupt(should_break)) return;
                auto [this_stump, this_metric]
                    = get_best_stump(feature_posneg, score_posneg,
                                     min_feature_value, max_feature_value,
                                     *metric_subsampled,
                                     workspace1, workspace2);
                if (interaction && detect_interrupt(should_break)) return;
                if (this_metric >= new_best_metric) {
                    new_best_metric = this_metric;
                    new_best_stump = Stump(j, this_stump);
                }
            }
            proposed_stump = new_best_stump;
        }

        std::vector<float> new_score = score;
        proposed_stump.add(X, feature_scales, new_score);

        metric_t new_metric;
        {
            PosNeg new_score_posneg(new_score, y);
            std::sort(new_score_posneg.pos.begin(), new_score_posneg.pos.end());
            std::sort(new_score_posneg.neg.begin(), new_score_posneg.neg.end());
            new_metric = metric(new_score_posneg);
        }
        if (new_metric >= best_metric) {
            learned_stumps.push_back(proposed_stump);
            score = std::move(new_score);
            best_metric = new_metric;
        } else {
            // Just add a null stump, so that we can keep track of what happened here.
            learned_stumps.emplace_back();
        }

        if (interaction) {
            #pragma omp critical
            {
                feed_pg(score);
            }
        }
    }
}

std::pair<StumpGivenFeature, metric_t>
    ExactBoostSingle::get_best_stump(const PosNeg& feature, const PosNeg& score,
                                     float min_feature_value, float max_feature_value,
                                     BaseMetric& metric,
                                     PosNeg& workspace1, PosNeg& workspace2) {
    assert(std::is_sorted(score.pos.cbegin(), score.pos.cend()));
    assert(std::is_sorted(score.neg.cbegin(), score.neg.cend()));

    IntervalStumpGivenFeature stump({min_feature_value, max_feature_value},
                                    {-1.0F, 1.0F},
                                    {-1.0F, 1.0F});
    std::array<std::pair<IntervalStumpGivenFeature, metric_t>, 8> subimages;

    for (;;) {
        Interval xi0 = stump.xi.fst();
        Interval xi1 = stump.xi.snd();
        Interval a0  = stump. a.fst();
        Interval a1  = stump. a.snd();
        Interval b0  = stump. b.fst();
        Interval b1  = stump. b.snd();

        {
            auto sep_pos = IntervalStumpGivenFeature::prepare(feature.pos, score.pos, workspace1.pos, xi0);
            auto sep_neg = IntervalStumpGivenFeature::prepare(feature.neg, score.neg, workspace1.neg, xi0);
            subimages[0] = {{xi0, a0, b0}, get_metric_upper_bound(feature, workspace1, sep_pos, sep_neg, {xi0, a0, b0}, metric, workspace2)};
            subimages[1] = {{xi0, a0, b1}, get_metric_upper_bound(feature, workspace1, sep_pos, sep_neg, {xi0, a0, b1}, metric, workspace2)};
            subimages[2] = {{xi0, a1, b0}, get_metric_upper_bound(feature, workspace1, sep_pos, sep_neg, {xi0, a1, b0}, metric, workspace2)};
            subimages[3] = {{xi0, a1, b1}, get_metric_upper_bound(feature, workspace1, sep_pos, sep_neg, {xi0, a1, b1}, metric, workspace2)};
        }
        {
            auto sep_pos = IntervalStumpGivenFeature::prepare(feature.pos, score.pos, workspace1.pos, xi1);
            auto sep_neg = IntervalStumpGivenFeature::prepare(feature.neg, score.neg, workspace1.neg, xi1);
            subimages[4] = {{xi1, a0, b0}, get_metric_upper_bound(feature, workspace1, sep_pos, sep_neg, {xi1, a0, b0}, metric, workspace2)};
            subimages[5] = {{xi1, a0, b1}, get_metric_upper_bound(feature, workspace1, sep_pos, sep_neg, {xi1, a0, b1}, metric, workspace2)};
            subimages[6] = {{xi1, a1, b0}, get_metric_upper_bound(feature, workspace1, sep_pos, sep_neg, {xi1, a1, b0}, metric, workspace2)};
            subimages[7] = {{xi1, a1, b1}, get_metric_upper_bound(feature, workspace1, sep_pos, sep_neg, {xi1, a1, b1}, metric, workspace2)};
        }
        std::shuffle(subimages.begin(), subimages.end(), rng);
        auto [max, _] = *std::max_element(subimages.cbegin(), subimages.cend(),
                                          [](const auto& lhs, const auto& rhs) {
                                              return lhs.second < rhs.second;
                                          });

        if (max.xi == stump.xi && max.a == stump.a && max.b == stump.b)
            break;

        stump = max;
    }

    metric_t best_metric = 0;
    StumpGivenFeature best_stump;
    for (float _xi = stump.xi.inf; _xi <= stump.xi.sup; _xi = std::nextafter(_xi, +INFINITY))
        for (float _a = stump.a.inf; _a <= stump.a.sup; _a = std::nextafter(_a, +INFINITY))
            for (float _b = stump.b.inf; _b <= stump.b.sup; _b = std::nextafter(_b, +INFINITY))
                if (metric_t this_metric = get_metric(feature, score, {_xi, _a, _b}, metric, workspace1);
                    this_metric >= best_metric) {
                    best_metric = this_metric;
                    best_stump = StumpGivenFeature(_xi, _a, _b);
                }
    return {best_stump, best_metric};
}

metric_t ExactBoostSingle::get_metric_upper_bound(const PosNeg& feature, const PosNeg& score,
                                                  const std::pair<std::vector<float>::const_iterator, std::vector<float>::const_iterator>& sep_pos,
                                                  const std::pair<std::vector<float>::const_iterator, std::vector<float>::const_iterator>& sep_neg,
                                                  const IntervalStumpGivenFeature& stump,
                                                  BaseMetric& metric,
                                                  PosNeg& workspace) {
    stump.apply(feature.pos, score.pos, sep_pos, workspace.pos, true);
    stump.apply(feature.neg, score.neg, sep_neg, workspace.neg, false);

    // margin:
    float abs = stump.a.inf <= stump.b.sup && stump.b.inf <= stump.a.sup
        ? 0.0F
        : std::min(std::abs(stump.a.inf - stump.b.sup),
                   std::abs(stump.b.inf - stump.a.sup));
    float margin_part = margin*(1.0F + abs*0.5F);
    for (float& s : workspace.pos) s -= margin_part;

    return metric(workspace);
}

metric_t ExactBoostSingle::get_metric(const PosNeg& feature, const PosNeg& score,
                                      const StumpGivenFeature& stump,
                                      BaseMetric& metric,
                                      PosNeg& workspace) {
    stump.apply(feature.pos, score.pos, workspace.pos);
    stump.apply(feature.neg, score.neg, workspace.neg);

    // margin:
    float margin_part = margin*(1.0F + std::abs(stump.b - stump.a)*0.5F);
    for (float& s : workspace.pos) s -= margin_part;

    return metric(workspace);
}

std::vector<float> ExactBoostSingle::predict(const std::vector<std::vector<float>>& X,
                                             const std::vector<float>& feature_scales) {
    const length_t n = X.front().size();

    std::vector<float> score = generate_random_score(n);
    for (const auto& stump : learned_stumps)
        stump.add(X, feature_scales, score);

    normalize01(score);

    return score;
}

std::vector<float> ExactBoostSingle::generate_random_score(length_t n) {
    std::vector<float> score(n);
    std::uniform_real_distribution dist(0.0F, 1.0F);
    std::generate(score.begin(), score.end(), [this, &dist]() { return dist(rng); });

    return score;
}

std::vector<std::vector<Stump>> ExactBoost::get_stumps() const {
    std::vector<std::vector<Stump>> stumps;
    std::transform(learned_estimators.begin(), learned_estimators.end(), std::back_inserter(stumps),
                   [](const ExactBoostSingle& es) { return es.learned_stumps; });
    return stumps;
}


ProgressBar::ProgressBar(unsigned int n_estimators, unsigned int n_rounds,
                         const BaseMetric& metric, const std::vector<uint8_t>& labels)
    : pg {
        indicators::option::BarWidth{50},
        indicators::option::Start{"["},
        indicators::option::End{"]"},
        indicators::option::ShowPercentage{true},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true},
    }
    , max_count(n_estimators*n_rounds)
    , progress(0)
    , last_update_time(std::chrono::steady_clock::now())
    , scores(n_estimators, std::vector<float>(labels.size(), 0))
    , metric(metric)
    , labels(labels)
    , n0(std::count(labels.cbegin(), labels.cend(), 0))
    , n1(std::count(labels.cbegin(), labels.cend(), 1))
{ }


ProgressBar::~ProgressBar() {
    if (progress == max_count)
        pg.set_progress(100.0F);
    pg.mark_as_completed();
}

void ProgressBar::set_score(unsigned int estimator_no, const std::vector<float>& score) {
    scores[estimator_no] = score;
}

void ProgressBar::update() {
    auto now = std::chrono::steady_clock::now();
    ++progress;

    const std::chrono::milliseconds update_delta{100};
    if (now - last_update_time <= update_delta)
        return;

    pg.set_progress(100.0F * (float)progress / (float)max_count);
    pg.set_option(indicators::option::PrefixText{
        fmt::format("  metric={:.2} ", evaluate_metric())
    });

    last_update_time = now;
}

void ProgressBar::finish() {
    pg.set_progress(100.0F);
}

ProgressBar::feed_func ProgressBar::get_feed_func(unsigned int estimator_no) {
    return [this, estimator_no](const std::vector<float>& score) {
        set_score(estimator_no, score);
        update();
    };
}

double ProgressBar::evaluate_metric() const {
    const length_t n = scores.front().size();

    std::vector<float> whole_score(n, 0);
    for (const auto& this_score : scores)
        for (length_t i = 0; i < n; ++i)
            whole_score[i] += this_score[i];

    float inv_n_scores = 1.0F / (float)scores.size();
    for (float& x : whole_score) x *= inv_n_scores;

    normalize01(whole_score);

    PosNeg score_posneg(whole_score, labels);
    std::sort(score_posneg.pos.begin(), score_posneg.pos.end());
    std::sort(score_posneg.neg.begin(), score_posneg.neg.end());
    metric_t metric_int = metric(score_posneg);

    return (double)metric_int / ((double)n0*(double)n1);
}
