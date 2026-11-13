# Pre-Trained Model Trade-Offs Across NLP, Vision, and Tabular Data

## Introduction

Selecting a pre-trained model requires balancing task quality with model size, inference speed, deployment constraints, and explainability. These trade-offs are especially important when systems must be efficient, dependable, auditable, or used in consequential settings. Accuracy alone is not a complete measure of trustworthiness.

This comparison covers two NLP encoders, two computer-vision classifiers, and one pre-trained tabular classifier. Their benchmark scores measure different tasks, so the numbers are compared **within domains**, not ranked across domains.

## Methodology

I selected models with source papers that report parameter counts and quantitative task results. For NLP, I chose BERT-base and DistilBERT rather than GPT or Gemini because the selected pair has publicly reported, comparable quality and CPU timing results. These are language-understanding encoders, not generative chat models, so the NLP conclusions apply to understanding tasks rather than text generation. For vision, I used ImageNet top-1 accuracy and model cost for MobileNetV2 and EfficientNet-B0; latency comes from the papers’ stated hardware and test configurations. For tabular classification, I selected TabPFN because it is pre-trained and evaluated on a published OpenML benchmark.

Parameter-only FP32 weight storage is estimated as four bytes per parameter. This is not the full memory needed to serve a model: runtime frameworks, activations, input length, batch size, and data context add memory. Published inference times are reported with their hardware, batch, and task conditions. They are not presented as a single cross-domain speed ranking.

## Decision Matrix

| Domain | Model | Size | Task metric | Reported speed | Additional considerations |
|---|---|---:|---|---|---|
| NLP | BERT-base | 110M parameters; about 440 MB FP32 weights | GLUE aggregate score: 79.5 | 668 seconds for a full STS-B development-set pass on CPU, batch size 1 | Higher GLUE score than DistilBERT in the cited study, with more parameters and longer CPU runtime. Encoder model, not a generative chat model. |
| NLP | DistilBERT | 66M parameters; about 264 MB FP32 weights | GLUE aggregate score: 77.0 | 410 seconds on the same CPU STS-B test | Smaller and faster in the cited test, with a lower aggregate GLUE score. Useful where deployment footprint matters. |
| Vision | MobileNetV2 | 3.4M parameters; about 14 MB FP32 weights; 300M multiply-adds | ImageNet top-1: 72.0% | 75 ms on one large core of a Google Pixel 1 using TensorFlow Lite | Designed for mobile and edge constraints. The device is old; benchmark again on target hardware. |
| Vision | EfficientNet-B0 | 5.3M parameters; about 21 MB FP32 weights; 0.39B FLOPs | ImageNet top-1: 77.1% | 13.4 ms TPUv3 / 38.1 ms V100, batch size 128 in a separate accelerator paper | Higher ImageNet accuracy in the cited results, with more computation. These latency figures do not directly compare with MobileNetV2’s Pixel CPU timing. |
| Tabular | TabPFN (v1 paper) | 25.82M parameters; about 103 MB FP32 weights | 18 small numerical OpenML datasets: mean ROC AUC 0.934 with 32-permutation ensembling; 0.932 without | No ensemble: 1.301 s CPU / 0.0519 s GPU per dataset. Ensemble: 37.59 s CPU / 0.6172 s GPU. | Pre-trained for small classification datasets. The benchmark used up to 1,000 training samples, 100 features, and 10 classes; results are weaker with categorical features or missing values. |

The matrix is also available as a standalone [SVG graphic](decision-matrix.svg).

## Analysis and Recommendations

### NLP

BERT-base is the stronger of the two on the cited GLUE aggregate score, while DistilBERT reduces parameters from 110M to 66M and takes less time on the paper’s CPU test. I would start with DistilBERT where serving footprint or response time is constrained, then compare it with BERT-base on the target task before accepting any quality loss. Both are text encoders; they should not be presented as general-purpose text generators.

### Computer vision

MobileNetV2 is a sensible candidate for mobile or edge deployments where model and compute budgets are tight. EfficientNet-B0 offers a higher ImageNet top-1 score in the cited results with a moderately larger parameter count and more operations. The reported latency settings differ, so the decision requires a fair benchmark on the actual device, input resolution, inference runtime, and batch size.

### Tabular data

TabPFN is attractive when the dataset is small, mostly numerical, and clean, and when a model that can use pre-training without per-dataset gradient training is valuable. Its benchmark limitations matter: the cited paper focuses on small datasets and reports weaker aggregate performance when categorical features or missing data are present. For larger or less constrained tabular problems, trained tree methods such as LightGBM or XGBoost are important baselines. They are not fixed pre-trained checkpoints: their size, speed, and quality depend on the fitted dataset and configuration. Tree-based methods can also provide more direct feature-level inspection, though that does not automatically make every prediction transparent.

### Explainability and trustworthiness

Model explanations should be evaluated separately from model performance. Attention maps, feature importance, or generated reasoning text can be useful inspection aids, but they should not automatically be treated as faithful evidence of the model’s internal reasoning. Explanation fidelity needs its own validation. A responsible evaluation should also examine calibration, robustness, safety, subgroup fairness, privacy, and human oversight as distinct concerns. No single score establishes trustworthiness.

## Classification Process

I classified each model by its input and intended task: text understanding, image classification, or structured tabular classification. I then sorted evidence into four decision dimensions: parameter/weight size, task-specific quality, speed under the source’s stated conditions, and deployment or interpretation constraints. I kept benchmark metrics separate because GLUE, ImageNet top-1 accuracy, and ROC AUC answer different questions. This avoids a misleading overall winner while still revealing patterns: distillation trades some language score for a smaller model; efficient vision networks target limited compute; and a pre-trained tabular transformer can perform strongly within a narrow data regime.

## Reflection on Learning

This comparison strengthened my understanding that model selection is a multi-criteria decision, not a search for the highest score. I paid closer attention to the benchmark protocol, hardware, batch size, and dataset limits because those details change what a speed or accuracy result means in deployment. Feedback on my prior portfolio also reinforced that explanation methods should not be confused with proof of a model’s reasoning. I now see trustworthiness as a set of separate checks—quality, calibration, robustness, safety, subgroup fairness, explanation fidelity, and human oversight—rather than a single performance number. For a real application, I would use published results to narrow candidates, then evaluate them on representative data with the intended users and deployment constraints in view.

## Conclusion

There is no universal best model among these examples. DistilBERT and MobileNetV2 emphasize reduced footprint; BERT-base and EfficientNet-B0 show stronger scores on their cited benchmarks at higher compute cost; TabPFN performs strongly on a small, clean tabular benchmark but has clear limits on data shape. The appropriate choice depends on the target task, acceptable latency and memory, data conditions, explainability needs, and the consequences of model error.

## Sources

1. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108). *NeurIPS EMC² Workshop*. Reports BERT-base and DistilBERT parameter counts, GLUE scores, and CPU timing.
2. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf). *CVPR*. Reports ImageNet top-1, parameters, multiply-adds, and Pixel 1 TensorFlow Lite timing.
3. Tan, M., & Le, Q. V. (2019). [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946). *ICML*. Reports EfficientNet-B0 ImageNet accuracy, parameter count, and FLOPs.
4. Li, S. et al. (2021). [Searching for Fast Model Families on Datacenter Accelerators](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Searching_for_Fast_Model_Families_on_Datacenter_Accelerators_CVPR_2021_paper.pdf). *CVPR*. Reports EfficientNet-B0 accelerator latency and specifies batch/hardware conditions.
5. Hollmann, N., Müller, S., Eggensperger, K., & Hutter, F. (2023). [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://openreview.net/pdf/a14bada70718d8e2f05879f7f5dd162a0adbe28c.pdf). *ICLR*. Reports model size, OpenML results, and CPU/GPU timings.

## Portfolio Placement

Suggested artifact title: **Pre-Trained Model Trade-Offs Across NLP, Vision, and Tabular Data**

Suggested caption: *A sourced decision matrix comparing model size, domain-specific benchmark quality, inference speed, deployment fit, and explainability considerations.*

For the wider portfolio, retain your own About Me section, skills list, and other work samples. Add alt text for the matrix and make sure the published page and source links are accessible to your instructor.


