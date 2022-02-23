#pragma once

#include <vector>
#include <random>
#include <chrono>
#include <indicators/block_progress_bar.hpp>

#include "metrics.hpp"


class ProgressBar {
    public:
        ProgressBar(unsigned int n_estimators, unsigned int n_rounds,
                    const BaseMetric& metric, const std::vector<uint8_t>& labels);
        ProgressBar(const ProgressBar& src) = delete;
        ProgressBar(const ProgressBar&& src) = delete;
        ProgressBar& operator=(const ProgressBar& src) = delete;
        ProgressBar& operator=(const ProgressBar&& src) = delete;
        ~ProgressBar();

        void set_score(unsigned int estimator_no, const std::vector<float>& score);
        void update();
        void finish();

        using feed_func = std::function<void(const std::vector<float>&)>;
        feed_func get_feed_func(unsigned int estimator_no);

    private:
        [[nodiscard]] double evaluate_metric() const;

        indicators::BlockProgressBar pg;
        size_t max_count;
        size_t progress;

        std::chrono::time_point<std::chrono::steady_clock> last_update_time;

        std::vector<std::vector<float>> scores;

        const BaseMetric& metric;
        const std::vector<uint8_t>& labels;

        const length_t n0;
        const length_t n1;
};


class ExactBoostSingle {
    public:
        ExactBoostSingle(const BaseMetric& metric, float margin,
                         unsigned int n_rounds,
                         double observation_subsampling, double feature_subsampling,
                         std::mt19937&& rng);
        ExactBoostSingle(const BaseMetric& metric, float margin,
                         unsigned int n_rounds,
                         double observation_subsampling, double feature_subsampling,
                         std::vector<Stump> learned_stumps,
                         std::mt19937&& rng);

        void fit(const std::vector<std::vector<float>>& X, const
                 std::vector<uint8_t>& y,
                 bool interaction,
                 const ProgressBar::feed_func& feed_pg,
                 const std::vector<float>& feature_scales, bool& should_break);
        [[nodiscard]] std::vector<float> predict(const std::vector<std::vector<float>>& X,
                                                 const std::vector<float>& feature_scales);

        std::vector<Stump> learned_stumps;

    private:
        [[nodiscard]] std::pair<StumpGivenFeature, metric_t>
            get_best_stump(const PosNeg& feature, const PosNeg& score,
                           float min_feature_value, float max_feature_value,
                           BaseMetric& metric,
                           PosNeg& workspace1, PosNeg& workspace2);

        [[nodiscard]] metric_t get_metric_upper_bound(const PosNeg& feature,
                                                      const PosNeg& score,
                                                      const std::pair<std::vector<float>::const_iterator, std::vector<float>::const_iterator>& sep_pos,
                                                      const std::pair<std::vector<float>::const_iterator, std::vector<float>::const_iterator>& sep_neg,
                                                      const IntervalStumpGivenFeature& stump,
                                                      BaseMetric& metric,
                                                      PosNeg& workspace);
        [[nodiscard]] metric_t get_metric(const PosNeg& feature,
                                          const PosNeg& score,
                                          const StumpGivenFeature& stump,
                                          BaseMetric& metric,
                                          PosNeg& workspace);

        [[nodiscard]] std::vector<float> generate_random_score(length_t n);

        const BaseMetric& metric;
        const float margin;
        const unsigned int n_rounds;
        const double observation_subsampling;
        const double feature_subsampling;

        std::mt19937 rng;
};


class ExactBoost {
    public:
        ExactBoost(const BaseMetric& metric, float margin,
                   unsigned int n_estimators, unsigned int n_rounds,
                   double observation_subsampling, double feature_subsampling);
        ExactBoost(const BaseMetric& metric, float margin,
                   unsigned int n_estimators, unsigned int n_rounds,
                   double observation_subsampling, double feature_subsampling,
                   const std::vector<float>& feature_scales,
                   const std::vector<std::vector<Stump>>& learned_stumps);

        void fit(const std::vector<std::vector<float>>& X, const std::vector<uint8_t>& y,
                 bool interaction);
        [[nodiscard]] std::vector<float> predict(const std::vector<std::vector<float>>& X);

        void fit(const xt::xtensor<float, 2>& X, const xt::xtensor<uint8_t, 1>& y,
                 bool interaction);
        [[nodiscard]] xt::xtensor<float, 1> predict(const xt::xtensor<float, 2>& X);

        [[nodiscard]] std::vector<std::vector<Stump>> get_stumps() const;

        const BaseMetric& metric;
        const float margin;
        const unsigned int n_estimators;
        const unsigned int n_rounds;
        const double observation_subsampling;
        const double feature_subsampling;

        std::vector<float> feature_scales;

        std::vector<ExactBoostSingle> learned_estimators;
};
