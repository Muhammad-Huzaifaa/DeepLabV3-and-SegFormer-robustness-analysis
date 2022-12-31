# DeepLabV3-and-SegFormer-robustness-analysis
In this project, we conducted a comparative analysis of
Segformer, a ViT-based semantic segmentation model, and
DeeplabV3+, a CNN-based semantic segmentation model,
using various image perturbations (e.g., image patch shuffling and removal) and noise addition (e.g., salt and pepper
noise). Segformer has been proven to be very efficient due
to its hierarchical encoder, and the spatial pyramid pooling in DeeplabV3+ allows it to compete with SOTA semantic segmentation models. Previous studies have compared
these models’ performances on the segmentation of natural images without any noise; however, the performance
of these models on perturbed and noisy images was unknown. Both models were trained on the ADE20k dataset
and tested on segmentation of the person class, which alleviates the effect of Segformer pre-training on ImageNet. Results suggest that Segformer can tolerate more perturbation
than DeeplabV3+ and can perform well on noisy images;
however, both models’ performance drops significantly as
we increase the noise.

