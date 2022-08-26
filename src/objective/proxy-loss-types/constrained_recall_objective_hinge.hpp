#include "../constrained_recall_objective.hpp"

namespace LightGBM {
    
    class ConstrainedRecallObjectiveHinge : public ConstrainedRecallObjective {
    public:
        
        explicit ConstrainedRecallObjectiveHinge(const Config &config): ConstrainedRecallObjective(config) {}

        double ComputePredictiveLoss(label_t label, double score) const override {

            if (abs(label) < 1e-5) // if (y_i == 0)
                return 0.;
            return score < proxy_margin_ ? proxy_margin_ - score : 0.;          // proxy_margin_ is the HORIZONTAL margin!

        }

        void GetGradients(const double *score, score_t *gradients, score_t *hessians) const override {
            /**
            * How much to shift the cross-entropy function (horizontally) to get
            * the target proxy_margin_ at x=0; i.e., f(0) = proxy_margin_
            */
            const double xent_horizontal_shift = log(exp(proxy_margin_) - 1);

            /**
            * NOTE
            *  - https://github.com/feedzai/fairgbm/issues/11
            *  - This value should be zero in order to optimize solely for TPR (Recall),
            *  as TPR considers only label positives (LPs) and ignores label negatives (LNs).
            *  - However, initial splits will have -inf information gain if the gradients
            *  of all LNs are 0;
            *  - Hence, we're adding a tiny positive weight to the gradient of all LNs;
            */
            const double label_negative_weight = 1e-2;

            #pragma omp parallel for schedule(static)
            for (data_size_t i = 0; i < num_data_; ++i) {
                // Proxy FNR (or proxy Recall) has no loss for label negative samples (they're ignored).
                if (abs(label_[i] - 1) < 1e-5) {  // if (y_i == 1)
                    
                    gradients[i] = (score_t) (score[i] < proxy_margin_ ? -1. : 0.);
                    hessians[i] = (score_t) 0.;

                    if (weights_ != nullptr) {
                        gradients[i] *= weights_[i];
                        hessians[i] *= weights_[i];
                    }

                }
                else {
                    // NOTE: https://github.com/feedzai/fairgbm/issues/11
                    //  - This whole else clause should not be needed to optimize for Recall,
                    //  as LNs have no influence on the FNR loss function or its (proxy-)gradient;
                    //  - However, passing a zero gradient to all LNs leads to weird early stopping
                    //  behavior from the `GBDT::Train` function;
                    //  - Adding this tiny weight to the gradient of LNs seems to fix the issue with
                    //  no (apparent) unintended consequences, as the gradient flowing is really small;
                    const double z = Constrained::sigmoid(score[i] + xent_horizontal_shift);
                    gradients[i] = (score_t) (label_negative_weight * z);
                    hessians[i] = (score_t) (label_negative_weight * z * (1. - z));
                }
            }
        }

    };
}