#pragma once

#include <memory>

#include "utils.hpp"


class BaseMetric {
    public:
        virtual metric_t operator()(const PosNeg& score) const = 0;
        [[nodiscard]] virtual std::unique_ptr<BaseMetric> with_subsampling(double observation_subsampling) const = 0;
};

class AUC : public BaseMetric {
    public:
        AUC();

        metric_t operator()(const PosNeg& score) const override;
        [[nodiscard]] std::unique_ptr<BaseMetric> with_subsampling(double observation_subsampling) const override;
};

class KS : public BaseMetric {
    public:
        KS();

        metric_t operator()(const PosNeg& score) const override;
        [[nodiscard]] std::unique_ptr<BaseMetric> with_subsampling(double observation_subsampling) const override;
};

class PaK : public BaseMetric {
    public:
        PaK(length_t k);

        metric_t operator()(const PosNeg& score) const override;
        [[nodiscard]] std::unique_ptr<BaseMetric> with_subsampling(double observation_subsampling) const override;

        length_t k;
};
