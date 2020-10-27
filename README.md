# AI Security Reading List
- [AI for IoT and Mobile](#ai-for-iot-and-mobile)
  * [Compression](#compression)
    + [Combining or Other](#combining-or-other)
    + [Distillation](#distillation)
    + [Factorization](#factorization)
    + [Pruning](#pruning)
    + [Quantization](#quantization)
  * [Misc](#misc)
- [Attacks and Defenses](#attacks-and-defenses)
  * [Adversarial Examples](#adversarial-examples)
    + [Attacks](#attacks)
    + [Defenses](#defenses)
  * [Backdoor](#backdoor)
    + [Attacks](#attacks-1)
    + [Defenses](#defenses-1)
  * [Inference](#inference)
    + [Attacks](#attacks-2)
    + [Defenses](#defenses-2)
  * [Poisoning](#poisoning)
    + [Attacks](#attacks-3)
    + [Defenses](#defenses-3)
- [Federated Learning](#federated-learning)
- [GAN and VAE](#gan-and-vae)
- [Interpretability and Attacks to New Scenario](#interpretability-and-attacks-to-new-scenario)
- [Multimodal](#multimodal)
- [SGX, TrustZone and Crypto](#sgx--trustzone-and-crypto)
- [Survey](#survey)
- [Other links](#other-links)

## AI for IoT and Mobile

### Compression

#### Combining or Other

- 2016, **ICLR**, [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/pdf/1510.00149)

- 2017, **SenSys**, [DeepIoT: Compressing Deep Neural Network Structures for Sensing Systems with a Compressor-Critic Framework](https://arxiv.org/pdf/1706.01215)

- 2018, **arXiv**, [To compress or not to compress: Understanding the Interactions between Adversarial Attacks and Neural Network Compression](https://arxiv.org/pdf/1810.00208)

- 2018, **ECCV**, [AMC: AutoML for Model Compression and Acceleration on Mobile Devices](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yihui_He_AMC_Automated_Model_ECCV_2018_paper.pdf)

- 2018, **ICLR**, [Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training](https://arxiv.org/pdf/1712.01887.pdf?utm_campaign=nathan.ai%20newsletter&utm_medium=email&utm_source=Revue%20newsletter)

- 2018, **SenSys**, [FastDeepIoT: Towards Understanding and Optimizing Neural Network Execution Time on Mobile and Embedded Devices](https://arxiv.org/pdf/1809.06970)

- 2019, **arXiv**, [A Programmable Approach to Model Compression](https://arxiv.org/pdf/1911.02497)

- 2019, **arXiv**, [Neural Network Distiller: A Python Package For DNN Compression Research](https://arxiv.org/pdf/1910.12232)

- 2019, **BigComp**, [Towards Robust Compressed Convolutional Neural Networks](https://ieeexplore.ieee.org/abstract/document/8679132/)

- 2019, **CVPR**, [FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_FBNet_Hardware-Aware_Efficient_ConvNet_Design_via_Differentiable_Neural_Architecture_Search_CVPR_2019_paper.pdf)

- 2019, **CVPR**, [HAQ: Hardware-Aware Automated Quantization with Mixed Precision](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.pdf)

- 2019, **CVPR**, [MnasNet: Platform-Aware Neural Architecture Search for Mobile](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.pdf)

- 2019, **NeurIPS**, [Positive-Unlabeled Compression on the Cloud](https://papers.nips.cc/paper/8525-positive-unlabeled-compression-on-the-cloud.pdf)

#### Distillation

- 2015, **NeurIPS**, [Distilling the Knowledge in a Neural Network](https://arxiv.xilesou.top/pdf/1503.02531)

- 2019, **arXiv**, [Adversarially Robust Distillation](https://arxiv.org/pdf/1905.09747)

- 2020, **AAAI**, [Adversarially Robust Distillation](https://arxiv.org/pdf/1905.09747)

- 2020, **AAAI**, [Ultrafast Video Attention Prediction with Coupled Knowledge Distillation](https://arxiv.xilesou.top/pdf/1904.04449)

#### Factorization

- 2019, **arXiv**, [Robust Sparse Regularization: Simultaneously Optimizing Neural Network Robustness and Compactness](https://arxiv.org/pdf/1905.13074)

#### Pruning

- 2017, **arXiv**, [Structural Compression of Convolutional Neural Networks Based on Greedy Filter Pruning](https://arxiv.xilesou.top/pdf/1705.07356)

- 2017, **arXiv**, [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/pdf/1710.01878)

- 2018, **arXiv**, [Dynamic Channel Pruning: Feature Boosting and Suppression](https://arxiv.org/pdf/1810.05331)

- 2019, **arXiv**, [Adversarial Neural Pruning with Latent Vulnerability Suppression](https://proceedings.icml.cc/static/paper_files/icml/2020/770-Paper.pdf)

- 2019, **arXiv**, [Localization-aware Channel Pruning for Object Detection](https://arxiv.org/pdf/1911.02237)

- 2019, **arXiv**, [Pruning by Explaining: A Novel Criterion for Deep Neural Network Pruning](https://arxiv.xilesou.top/pdf/1912.08881)

- 2019, **arXiv**, [Pruning from Scratch](https://arxiv.xilesou.top/pdf/1909.12579)

- 2019, **arXiv**, [Selective Brain Damage: Measuring the Disparate Impact of Model Pruning](https://arxiv.xilesou.top/pdf/1911.05248)

- 2019, **arXiv**, [Structured Pruning of Large Language Models](https://arxiv.xilesou.top/pdf/1910.04732)

- 2019, **arXiv**, [Towards Compact and Robust Deep Neural Networks](https://arxiv.xilesou.top/pdf/1906.06110)

- 2019, **ICCV**, [Adversarial Robustness vs. Model Compression, or Both](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ye_Adversarial_Robustness_vs._Model_Compression_or_Both_ICCV_2019_paper.pdf)

- 2019, **ICLR**, [Rethinking the Value of Network Pruning](https://arxiv.org/pdf/1810.05270)

- 2019, **ICLR**, [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/pdf/1803.03635)

- 2019, **ICONIP**, [Self-Adaptive Network Pruning](https://arxiv.org/pdf/1910.08906)

- 2019, **NeurIPS**, [Network Pruning via Transformable Architecture Search](https://arxiv.org/pdf/1905.09717)

- 2020, **AAAI**, [AutoCompress: An Automatic DNN Structured Pruning Framework for Ultra-High Compression Rates](https://www.researchgate.net/profile/Zhiyuan_Xu9/publication/334316382_AutoSlim_An_Automatic_DNN_Structured_Pruning_Framework_for_Ultra-High_Compression_Rates/links/5ddf9aab4585159aa44f1634/AutoSlim-An-Automatic-DNN-Structured-Pruning-Framework-for-Ultra-High-Compression-Rates.pdf)

- 2020, **AAAI**, [PCONV: The Missing but Desirable Sparsity in DNN Weight Pruning for Real-time Execution on Mobile Devices](https://arxiv.xilesou.top/pdf/1909.05073)

- 2020, **ASPLOS**, [PatDNN: Achieving Real-Time DNN Execution on Mobile Devices with Pattern-based Weight Pruning](https://arxiv.xilesou.top/pdf/2001.00138)

- 2020, **ICLR**, [Comparing Fine-tuning and Rewinding in Neural Network Pruning]()

#### Quantization

- 2018, **arXiv**, [Combinatorial Attacks on Binarized Neural Networks](https://arxiv.org/pdf/1810.03538)

- 2018, **CVPR**, [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf)

- 2018, **ICLR**, [Attacking Binarized Neural Networks](https://arxiv.org/pdf/1711.00449)

- 2019, **arXiv**, [Defensive Quantization: When Efficiency Meets Robustness](https://arxiv.org/pdf/1904.08444)

- 2019, **arXiv**, [Impact of Low-bitwidth Quantization on the Adversarial Robustness for Embedded Neural Networks](https://arxiv.org/pdf/1909.12741)

- 2019, **arXiv**, [Model Compression with Adversarial Robustness: A Unified Optimization Framework](https://papers.nips.cc/paper/8410-model-compression-with-adversarial-robustness-a-unified-optimization-framework.pdf)

- 2019, **ICLR**, [Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets](https://arxiv.org/pdf/1903.05662)

- 2020, **MLSys**, [Memory-Driven Mixed Low Precision Quantization For Enabling Deep Network Inference On Microcontrollers](https://arxiv.xilesou.top/pdf/1905.13082)

- 2020, **MLSys**, [Searching for Winograd-aware Quantized Networks](https://arxiv.xilesou.top/pdf/2002.10711)

- 2020, **MLSys**, [Trained Quantization Thresholds for Accurate and Efficient Fixed-Point Inference of Deep Neural Networks](https://proceedings.mlsys.org/static/paper_files/mlsys/2020/71-Paper.pdf)

### Misc

- 2016, **SenSys**, [Sparsification and Separation of Deep Learning Layers for Constrained Resource Inference on Wearables](http://discovery.ucl.ac.uk/1535346/1/main%20%283%29.pdf)

- 2017, **arXiv**, [Mobilenets: Efficient convolutional neural networks for mobile vision applications](https://arxiv.org/pdf/1704.04861)

- 2017, **ICLR**, [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and less than 0.5MB model size](https://arxiv.org/pdf/1602.07360)

- 2018, **ECCV**, [PIRM Challenge on Perceptual Image Enhancement on Smartphones: Report](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Ignatov_PIRM_Challenge_on_Perceptual_Image_Enhancement_on_Smartphones_Report_ECCVW_2018_paper.pdf)

- 2018, **mobicom**, [FoggyCache : Cross-Device Approximate Computation Reuse](http://www.cs.yale.edu/homes/guo-peizhen/files/foggycache-mobicom18.pdf)

- 2019, **arXiv**, [Characterizing the Deep Neural Networks Inference Performance of Mobile Applications](https://arxiv.org/pdf/1909.04783)

- 2019, **arXiv**, [Confidential Deep Learning: Executing Proprietary Models on Untrusted Devices](https://arxiv.org/pdf/1908.10730)

- 2019, **arXiv**, [On-Device Neural Net Inference with Mobile GPUs](https://arxiv.org/pdf/1907.01989)

- 2019, **HPCA**, [Machine Learning at Facebook: Understanding Inference at the Edge](https://research.fb.com/wp-content/uploads/2018/12/Machine-Learning-at-Facebook-Understanding-Inference-at-the-Edge.pdf)

- 2019, **WWW**, [A First Look at Deep Learning Apps on Smartphones](https://dl.acm.org/citation.cfm?id=3313591)

## Attacks and Defenses

### Adversarial Examples

#### Attacks

- 2016, **Euro S&P**, [The Limitations of Deep Learning in Adversarial Settings](https://arxiv.org/pdf/1511.07528.pdf&xid=25657,15700023,15700124,15700149,15700186,15700191,15700201,15700237,15700242)

- 2016, **ICLR**, [Adversarial Manipulation of Deep Representations](https://arxiv.org/pdf/1511.05122)

- 2017, **AISec**, [Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods](https://arxiv.xilesou.top/pdf/1705.07263)

- 2017, **AsiaCCS**, [Practical Black-Box Attacks against Machine Learning](https://dl.acm.org/doi/pdf/10.1145/3052973.3053009)

- 2017, **S&P**, [Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/pdf/1608.04644)

- 2018, **arXiv**, [Are adversarial examples inevitable](https://arxiv.org/pdf/1809.02104)

- 2018, **CVPR**, [Robust Physical-World Attacks on Deep Learning Visual Classification](http://openaccess.thecvf.com/content_cvpr_2018/papers/Eykholt_Robust_Physical-World_Attacks_CVPR_2018_paper.pdf)

- 2018, **ICLR**, [Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models](https://arxiv.org/pdf/1712.04248)

- 2018, **ICML**, [Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples](https://arxiv.org/pdf/1802.00420)

- 2018, **KDD**, [Adversarial Attacks on Neural Networks for Graph Data](https://arxiv.org/pdf/1805.07984)

- 2018, **USENIX**, [With Great Training Comes Great Vulnerability: Practical Attacks against Transfer Learning](https://www.usenix.org/system/files/conference/usenixsecurity18/sec18-wang.pdf)

- 2019, **arXiv**, [Adversarial Examples Are a Natural Consequence of Test Error in Noise](https://arxiv.org/pdf/1901.10513)

- 2019, **arXiv**, [Adversarial Examples Are Not Bugs, They Are Features](https://arxiv.org/pdf/1905.02175)

- 2019, **arXiv**, [Batch Normalization is a Cause of Adversarial Vulnerability](https://arxiv.org/pdf/1905.02161)

- 2019, **arXiv**, [WITCHcraft: Efficient PGD attacks with random step size](https://arxiv.org/pdf/1911.07989)

- 2019, **GECCO**, [GenAttack: practical black-box attacks with gradient-free optimization](https://arxiv.xilesou.top/pdf/1805.11090)

- 2019, **ICML**, [Revisiting Adversarial Risk](http://proceedings.mlr.press/v89/suggala19a/suggala19a.pdf)

- 2019, **NDSS**, [TEXTBUGGER: Generating Adversarial Text Against Real-world Applications](https://arxiv.org/pdf/1812.05271)

- 2020, **arXiv**, [Feature Purification: How Adversarial Training Performs Robust Deep Learning](https://arxiv.org/pdf/2005.10190)

- 2020, **arXiv**, [On Adaptive Attacks to Adversarial Example Defenses](https://arxiv.org/pdf/2002.08347)

- 2020, **arXiv**, [Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks](https://arxiv.org/pdf/2003.01690)

- 2020, **arXiv**, [Towards Feature Space Adversarial Attack](https://arxiv.org/pdf/2004.12385)

- 2020, **CCS**, [A Tale of Evil Twins: Adversarial Inputs versus Poisoned Models](https://arxiv.org/pdf/1911.01559)

- 2020, **CVPR**, [High-Frequency Component Helps Explain the Generalization of Convolutional Neural Networks](https://arxiv.org/abs/1905.13545)

- 2020, **ICLR**, [A Target-Agnostic Attack on Deep Models: Exploiting Security Vulnerabilities of Transfer Learning](https://arxiv.xilesou.top/pdf/1904.04334)

- 2020, **S&P**, [Intriguing Properties of Adversarial ML Attacks in the Problem Space](https://arxiv.org/pdf/1911.02142)

- 2020, **USENIX**, [Adversarial Preprocessing: Understanding and Preventing Image-Scaling Attacks in Machine Learning](https://www.usenix.org/system/files/sec20fall_quiring_prepub.pdf)

- 2020, **USENIX**, [Hybrid Batch Attacks: Finding Black-box Adversarial Examples with Limited Queries](https://arxiv.org/pdf/1908.07000)

#### Defenses

- 2015, **arXiv**, [Analysis of classifiers¡¯ robustness to adversarial perturbations](https://link.springer.com/article/10.1007/s10994-017-5663-3)

- 2016, **ICLR**, [Learning with a Strong Adversary](https://arxiv.org/pdf/1511.03034)

- 2016, **S&P**, [Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks](https://arxiv.org/pdf/1511.04508)

- 2017, **arXiv**, [Extending Defensive Distillation](https://arxiv.org/pdf/1705.05264)

- 2017, **arXiv**, [The Space of Transferable Adversarial Examples](https://arxiv.org/pdf/1704.03453.pdf?source=post_page---------------------------)

- 2017, **ICLR**, [Adversarial Machine Learning at Scale](https://arxiv.org/pdf/1611.01236)

- 2017, **ICML**, [Parseval Networks: Improving Robustness to Adversarial Examples](https://arxiv.org/pdf/1704.08847)

- 2018, **AAAI**, [Improving the adversarial robustness and interpretability of deep neural networks by regularizing their input gradients](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17337/15866)

- 2018, **arXiv**, [Adversarial Logit Pairing](https://arxiv.org/pdf/1803.06373)

- 2018, **arXiv**, [Adversarially Robust Training through Structured Gradient Regularization](https://arxiv.org/pdf/1805.08736)

- 2018, **arXiv**, [Gradient Adversarial Training of Neural Networks](https://arxiv.org/pdf/1806.08028)

- 2018, **arXiv**, [The Taboo Trap: Behavioural Detection of Adversarial Samples](https://arxiv.org/pdf/1811.07375)

- 2018, **CVPR**, [Partial Transfer Learning with Selective Adversarial Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cao_Partial_Transfer_Learning_CVPR_2018_paper.pdf)

- 2018, **ECCV**, [Is robustness the cost of accuracy¨Ca comprehensive study on the robustness of 18 deep image classification models](http://openaccess.thecvf.com/content_ECCV_2018/papers/Dong_Su_Is_Robustness_the_ECCV_2018_paper.pdf)

- 2018, **ICLR Workshop**, [Attacking the Madry Defense Model with L1-based Adversarial Examples](https://arxiv.org/pdf/1710.10733)

- 2018, **ICLR**, [Ensemble Adversarial Training: Attacks and Defenses](https://arxiv.org/pdf/1705.07204.pdf))

- 2018, **ICLR**, [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/pdf/1710.09412.pdf))

- 2018, **ICLR**, [Stochastic Activation Pruning for Robust Adversarial Defense](https://arxiv.org/pdf/1803.01442)

- 2018, **ICLR**, [Thermometer Encoding: One Hot Way To Resist Adversarial Examples](https://openreview.net/pdf?id=S18Su--CW)

- 2018, **ICML**, [An Optimal Control Approach to Deep Learning and Applications to Discrete-Weight Neural Networks](https://arxiv.org/pdf/1803.01299)

- 2018, **NeurIPS SECML**, [Evaluating and Understanding the Robustness of Adversarial Logit Pairing](https://arxiv.org/pdf/1807.10272.pdf))

- 2018, **NeurIPS SECML**, [Logit Pairing Methods Can Fool Gradient-Based Attacks](https://arxiv.org/pdf/1810.12042)

- 2018, **NeurIPS**, [Adversarially Robust Generalization Requires More Data](http://papers.nips.cc/paper/7749-adversarially-robust-generalization-requires-more-data.pdf)

- 2018, **NeurIPS**, [Sparse DNNs with Improved Adversarial Robustness](https://papers.nips.cc/paper/7308-sparse-dnns-with-improved-adversarial-robustness.pdf)

- 2019, **arXiv**, [Better the Devil you Know: An Analysis of Evasion Attacks using Out-of-Distribution Adversarial Examples](https://arxiv.org/pdf/1905.01726)

- 2019, **arXiv**, [Defending Against Misclassification Attacks in Transfer Learning](https://arxiv.org/pdf/1908.11230)

- 2019, **arXiv**, [On Evaluating Adversarial Robustness](https://arxiv.org/pdf/1902.06705)

- 2019, **arXiv**, [Scaleable input gradient regularization for adversarial robustness](https://arxiv.org/pdf/1905.11468)

- 2019, **arXiv**, [Sitatapatra: Blocking the Transfer of Adversarial Samples](https://arxiv.org/pdf/1901.08121)

- 2019, **arXiv**, [Stateful Detection of Black-Box Adversarial Attacks](https://arxiv.org/pdf/1907.05587)

- 2019, **arXiv**, [Towards Compact and Robust Deep Neural Networks](https://arxiv.xilesou.top/pdf/1906.06110)

- 2019, **arXiv**, [Towards Deep Learning Models Resistant to Adversarial Attacks ](https://arxiv.org/pdf/1706.06083)

- 2019, **arXiv**, [Transfer of Adversarial Robustness Between Perturbation Types](https://arxiv.org/pdf/1905.01034)

- 2019, **arXiv**, [Understanding Adversarial Robustness: The Trade-off between Minimum and Average Margin](https://arxiv.org/pdf/1907.11780)

- 2019, **arXiv**, [Using Honeypots to Catch Adversarial Attacks on Neural Networks](https://arxiv.org/pdf/1904.08554)

- 2019, **arXiv**, [Using Pre-Training Can Improve Model Robustness and Uncertainty](https://arxiv.org/pdf/1901.09960)

- 2019, **arXiv**, [What it Thinks is Important is Important: Robustness Transfers through Input Gradients](https://arxiv.xilesou.top/pdf/1912.05699)

- 2019, **CVPR**, [Disentangling Adversarial Robustness and Generalization](https://arxiv.org/abs/1812.00740)

- 2019, **CVPR**, [Feature Denoising for Improving Adversarial Robustness](https://arxiv.org/pdf/1812.03411)

- 2019, **ICCV**, [Bilateral Adversarial Training: Towards Fast Training of More Robust Models Against Adversarial Attacks](https://arxiv.org/abs/1811.10716)

- 2019, **ICLR**, [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.xilesou.top/pdf/1903.12261)

- 2019, **ICLR**, [L2-Nonexpansive Neural Networks](https://arxiv.org/abs/1802.07896)

- 2019, **ICLR**, [Robustness May Be at Odds with Accuracy](https://arxiv.xilesou.top/pdf/1805.12152.pdf,)

- 2019, **ICLR**, [Towards the first adversarially robust neural network model on MNIST](https://arxiv.org/abs/1805.09190)

- 2019, **ICLR**, [Training for Faster Adversarial Robustness Verification via Inducing ReLU Stability](https://arxiv.xilesou.top/pdf/1809.03008)

- 2019, **ICML**, [Theoretically Principled Trade-off between Robustness and Accuracy](https://arxiv.org/pdf/1901.08573)

- 2019, **NDSS**, [NIC: Detecting Adversarial Samples with Neural Network Invariant Checking](https://www.cs.purdue.edu/homes/ma229/papers/NDSS19.pdf)

- 2019, **NeurIPS**, [Adversarial Training and Robustness for Multiple Perturbations](http://papers.nips.cc/paper/8821-adversarial-training-and-robustness-for-multiple-perturbations.pdf)

- 2019, **NeurIPS**, [Adversarial Training for Free](https://arxiv.xilesou.top/pdf/1904.12843)

- 2019, **NeurIPS**, [Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers](https://arxiv.org/abs/1906.04584)

- 2019, **NeurIPS**, [You only propagate once: Painless adversarial training using maximal principle](http://papers.nips.cc/paper/8316-you-only-propagate-once-accelerating-adversarial-training-via-maximal-principle)

- 2019, **PMLR**, [Transferable Adversarial Training: A General Approach to Adapting Deep Classifiers](http://proceedings.mlr.press/v97/liu19b/liu19b.pdf)

- 2019, **USENIX**, [Improving Robustness of ML Classifiers against Realizable Evasion Attacks Using Conserved Features](https://www.usenix.org/system/files/sec19-tong.pdf)

- 2020, **AAAI**, [Universal Adversarial Training](http://openaccess.thecvf.com/content_cvpr_2017/html/Moosavi-Dezfooli_Universal_Adversarial_Perturbations_CVPR_2017_paper.html)

- 2020, **arXiv**, [Adversarially-Trained Deep Nets Transfer Better](https://arxiv.org/pdf/2007.05869.pdf(%5Bpaper%5D(https://arxiv.org/pdf/2007.05869.pdf))

- 2020, **arXiv**, [Do Adversarially Robust ImageNet Models Transfer Better](https://arxiv.org/pdf/2007.08489)

- 2020, **arXiv**, [Improving the Adversarial Robustness of Transfer Learning via Noisy Feature Distillation](https://arxiv.org/abs/2002.02998)

- 2020, **arXiv**, [One Neuron to Fool Them All](https://arxiv.org/pdf/2003.09372)

- 2020, **arXiv**, [Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks](https://arxiv.org/pdf/2003.01690)

- 2020, **ICLR**, [Adversarially robust transfer learning](https://arxiv.xilesou.top/pdf/1905.08232)

- 2020, **ICLR**, [Fast is better than free: Revisiting adversarial training](https://arxiv.org/pdf/2001.03994)

- 2020, **KDD**, [An Embarrassingly Simple Approach for Trojan Attack in Deep Neural Networks](https://dl.acm.org/doi/abs/10.1145/3394486.3403064)

- 2020, **USENIX**, [TEXTSHIELD: Robust Text Classification Based on Multimodal Embedding and Neural Machine Translation](https://nesa.zju.edu.cn/download/TEXTSHIELD%20Robust%20Text%20Classification%20Based%20on%20Multimodal%20Embedding%20and%20Neural%20Machine%20Translation.pdf)

### Backdoor

#### Attacks

- 2017, **arXiv**, [Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning](https://arxiv.xilesou.top/pdf/1712.05526)

- 2017, **CNS**, [Backdoor Attacks against Learning Systems](https://par.nsf.gov/servlets/purl/10066467)

- 2018, **CCS**, [Model-Reuse Attacks on Deep Learning Systems](https://arxiv.xilesou.top/pdf/1812.00483)

- 2018, **CoRR**, [Backdoor Embedding in Convolutional Neural Network Models via Invisible Perturbation](https://arxiv.xilesou.top/pdf/1808.10307)

- 2018, **NDSS**, [Trojaning Attack on Neural Networks](https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2782&context=cstech)

- 2019, **Access**, [BadNets: Evaluating Backdooring Attacks on Deep Neural Networks](https://ieeexplore.ieee.xilesou.top/iel7/6287639/8600701/08685687.pdf)

- 2019, **arXiv**, [Bypassing Backdoor Detection Algorithms in Deep Learning](https://arxiv.xilesou.top/pdf/1905.13409)

- 2019, **arXiv**, [Invisible Backdoor Attacks Against Deep Neural Networks](https://arxiv.xilesou.top/pdf/1909.02742)

- 2019, **arXiv**, [Label-Consistent Backdoor Attacks](https://arxiv.org/pdf/1912.02771/)

- 2019, **arXiv**, [Programmable Neural Network Trojan for Pre-Trained Feature Extractor](https://arxiv.org/pdf/1901.07766)

- 2019, **CCS**, [Regula Sub-rosa: Latent Backdoor Attacks on Deep Neural Networks](https://arxiv.xilesou.top/pdf/1905.10447)

- 2019, **Thesis**, [Exploring the Landscape of Backdoor Attacks on Deep Neural Network Models](https://dspace.mit.edu/bitstream/handle/1721.1/123127/1128278987-MIT.pdf?sequence=1&isAllowed=y)

- 2020, **AAAI**, [Hidden Trigger Backdoor Attacks](https://arxiv.org/pdf/1910.00033)

- 2020, **arXiv**, [Backdoor Attacks against Transfer Learning with Pre-trained Deep Learning Models](https://arxiv.xilesou.top/pdf/2001.03274)

- 2020, **arXiv**, [Clean-Label Backdoor Attacks on Video Recognition Models](https://arxiv.org/pdf/2003.03030)

- 2020, **arXiv**, [Piracy Resistant Watermarks for Deep Neural Networks](https://arxiv.org/pdf/1910.01226)

#### Defenses

- 2017, **ICCD**, [Neural Trojans](https://arxiv.xilesou.top/pdf/1710.00942)

- 2018, **arXiv**, [Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering](https://arxiv.xilesou.top/pdf/1811.03728)

- 2018, **arXiv**, [SentiNet: Detecting Physical Attacks Against Deep Learning Systems](https://arxiv.xilesou.top/pdf/1812.00292)

- 2018, **NeruIPS**, [Spectral Signatures in Backdoor Attacks](https://papers.nips.cc/paper/8024-spectral-signatures-in-backdoor-attacks.pdf)

- 2018, **RAID**, [Fine-Pruning Defending Against Backdooring Attacks on Deep Neural Network](https://arxiv.xilesou.top/pdf/1805.12185)

- 2019, **ACSAC**, [STRIP: a defence against trojan attacks on deep neural networks](https://arxiv.xilesou.top/pdf/1902.06531)

- 2019, **arXiv**, [NeuronInspect: Detecting Backdoors in Neural Networks via Output Explanations](https://arxiv.xilesou.top/pdf/1911.07399)

- 2019, **arXiv**, [TABOR: A Highly Accurate Approach to Inspecting and Restoring Trojan Backdoors in AI Systems](https://arxiv.xilesou.top/pdf/1908.01763)

- 2019, **CCS**, [ABS: Scanning Neural Networks for Back-doors by Artificial Brain Stimulation](https://dl.acm.org/doi/pdf/10.1145/3319535.3363216)

- 2019, **NeurIPS**, [Defending Neural Backdoors via Generative Distribution Modeling](http://papers.nips.cc/paper/9550-defending-neural-backdoors-via-generative-distribution-modeling.pdf)

- 2019, **Online**, [DeepInspect: A Black-box Trojan Detection and Mitigation Framework for Deep Neural Networks](http://www.aceslab.org/sites/default/files/DeepInspect.pdf)

- 2019, **S&P**, [Neural cleanse: Identifying and mitigating backdoor attacks in neural networks](https://people.cs.vt.edu/vbimal/publications/backdoor-sp19.pdf)

### Inference

#### Attacks

- 2017, **S&P**, [Membership Inference Attacks Against Machine Learning Models](https://arxiv.xilesou.top/pdf/1610.05820)

- 2018, **CCS**, [Property Inference Attacks on Fully Connected Neural Networks using Permutation Invariant Representations](http://youngwei.com/pdf/PermuteInvariance.pdf)

- 2019, **CCS**, [Privacy Risks of Securing Machine Learning Models against Adversarial Examples](https://arxiv.xilesou.top/pdf/1905.10291)

- 2019, **S&P**, [Comprehensive Privacy Analysis of Deep Learning: Passive and Active White-box Inference Attacks against Centralized and Federated Learning](https://gfsoso.99lb.net/sci-hub.html)

- 2020, **NDSS**, [CloudLeak: Large-Scale Deep Learning Models Stealing Through Adversarial Examples](http://jin.ece.ufl.edu/papers/NDSS2020_CloudLeak.pdf)

- 2020, **USENIX**, [Updates-Leak: Data Set Inference and Reconstruction Attacks in Online Learning](https://arxiv.org/pdf/1904.01067)

#### Defenses

- 2018, **arXiv**, [PRIVADO: Practical and Secure DNN Inference with Enclaves](https://gfsoso.99lb.net/sci-hub.html)

- 2018, **arXiv**, [YerbaBuena: Securing Deep Learning Inference Data via Enclave-based Ternary Model Partitioning](https://www.researchgate.net/profile/Ankita_Lamba/publication/326171835_Securing_Input_Data_of_Deep_Learning_Inference_Systems_via_Partitioned_Enclave_Execution/links/5b75c09092851ca65064df4e/Securing-Input-Data-of-Deep-Learning-Inference-Systems-via-Partitioned-Enclave-Execution.pdf)

- 2019, **CCS**, [MemGuard: Defending against Black-Box Membership Inference Attacks via Adversarial Examples](https://arxiv.xilesou.top/pdf/1909.10594)

- 2019, **mobicom**, [Occlumency: Privacy-preserving Remote Deep-learning Inference Using SGX](https://gfsoso.99lb.net/sci-hub.html)

- 2019, **S&P**, [Certified Robustness to Adversarial Examples with Differential Privacy](https://arxiv.xilesou.top/pdf/1802.03471)

- 2019, **SOSP**, [Privacy accounting and quality control in the sage differentially private ML platform](https://arxiv.xilesou.top/pdf/1909.01502)

### Poisoning

#### Attacks

- 2018, **NeurIPS**, [Poison Frogs Targeted Clean-Label Poisoning Attacks on Neural Networks](http://papers.nips.cc/paper/7849-poison-frogs-targeted-clean-label-poisoning-attacks-on-neural-networks)

- 2018, **USENIX**, [When Does Machine Learning FAIL Generalized Transferability for Evasion and Poisoning Attacks](https://www.usenix.org/conference/usenixsecurity18/presentation/suciu)

- 2020, **S&P**, [Humpty Dumpty: Controlling Word Meanings via Corpus Poisoning](https://arxiv.org/pdf/2001.04935)

#### Defenses

- 2019, **arXiv**, [Robust Graph Neural Network Against Poisoning Attacks via Transfer Learning](https://arxiv.xilesou.top/pdf/1908.07558)

## Federated Learning

- 2018, **arXiv**, [How To Backdoor Federated Learning](https://arxiv.xilesou.top/pdf/1807.00459)

- 2018, **arXiv**, [Mitigating Sybils in Federated Learning Poisoning](https://arxiv.xilesou.top/pdf/1808.04866)

- 2019, **arXiv**, [Can You Really Backdoor Federated Learning](https://arxiv.xilesou.top/pdf/1911.07963)

- 2019, **arXiv**, [Deep Leakage from Gradients](https://papers.nips.cc/paper/9617-deep-leakage-from-gradients.pdf)

- 2019, **arXiv**, [On Safeguarding Privacy and Security in the Framework of Federated Learning](https://arxiv.xilesou.top/pdf/1909.06512)

- 2019, **ICLR**, [Analyzing Federated Learning through an Adversarial Lens](https://arxiv.xilesou.top/pdf/1811.12470)

## GAN and VAE

- 2014, **ICLR**, [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf?source=post_page---------------------------)

- 2014, **NeurIPS**, [Generative Adversarial Nets](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

- 2016, **ICLR**, [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.xilesou.top/pdf/1511.06434.pdf%C3)

- 2016, **ICML**, [Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/pdf/1512.09300.pdf?source=post_page---------------------------)

- 2016, **NeurIPS**, [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](http://papers.nips.cc/paper/6399-infogan-interpretable-representation-learning-by-information-maximizing-generative-adversarial-nets.pdf)

- 2017, **arXiv**, [BEGAN: Boundary Equilibrium Generative Adversarial Networks](https://arxiv.xilesou.top/pdf/1703.10717)

- 2017, **NeurIPS**, [Improved Training of Wasserstein GANs](https://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans.pdf)

- 2017, **NeurIPS**, [Wasserstein GAN](https://arxiv.xilesou.top/pdf/1701.07875.pdf%20http://arxiv.org/abs/1701.07875)

- 2020, **CVPR**, [CNN-generated images are surprisingly easy to spot... for now](https://arxiv.xilesou.top/pdf/1912.11035)

## Interpretability and Attacks to New Scenario

- 2018, **arXiv**, [How To Backdoor Federated Learning](https://arxiv.xilesou.top/pdf/1807.00459)

- 2018, **arXiv**, [Mitigating Sybils in Federated Learning Poisoning](https://arxiv.xilesou.top/pdf/1808.04866)

- 2019, **arXiv**, [Can You Really Backdoor Federated Learning](https://arxiv.xilesou.top/pdf/1911.07963)

- 2019, **arXiv**, [Deep Leakage from Gradients](https://papers.nips.cc/paper/9617-deep-leakage-from-gradients.pdf)

- 2019, **arXiv**, [On Safeguarding Privacy and Security in the Framework of Federated Learning](https://arxiv.xilesou.top/pdf/1909.06512)

- 2019, **ICLR**, [Analyzing Federated Learning through an Adversarial Lens](https://arxiv.xilesou.top/pdf/1811.12470)

- 2020, **USENIX**, [Interpretable Deep Learning under Fire](https://arxiv.xilesou.top/pdf/1812.00891)

## Multimodal

- 2019, **arXiv**, [Supervised Multimodal Bitransformers for Classifying Images and Text](https://arxiv.org/abs/1909.02950)

- 2019, **arXiv**, [VisualBERT: A Simple and Performant Baseline for Vision and Language](https://arxiv.org/pdf/1908.03557)

- 2019, **TACL**, [Trick Me If You Can: Human-in-the-Loop Generation of Adversarial Examples for Question Answering](https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00279)

- 2020, **AAAI**, [Is BERT Really Robust: A Strong Baseline for Natural Language Attack on Text Classification and Entailment](https://zhijing-jin.com/files/papers/Is_BERT_Jin2020AAAI.pdf)

## SGX, TrustZone and Crypto

- 2018, **arXiv**, [PRIVADO: Practical and Secure DNN Inference with Enclaves](https://gfsoso.99lb.net/sci-hub.html)

- 2018, **arXiv**, [StreamBox-TZ: Secure Stream Analytics at the Edge with TrustZone](https://www.usenix.org/system/files/atc19-park-heejin.pdf)

- 2018, **arXiv**, [YerbaBuena: Securing Deep Learning Inference Data via Enclave-based Ternary Model Partitioning](https://www.researchgate.net/profile/Ankita_Lamba/publication/326171835_Securing_Input_Data_of_Deep_Learning_Inference_Systems_via_Partitioned_Enclave_Execution/links/5b75c09092851ca65064df4e/Securing-Input-Data-of-Deep-Learning-Inference-Systems-via-Partitioned-Enclave-Execution.pdf)

- 2019, **arXiv**, [Confidential Deep Learning: Executing Proprietary Models on Untrusted Devices](https://arxiv.org/pdf/1908.10730)

- 2019, **arXiv**, [Let the Cloud Watch Over Your IoT File Systems](https://arxiv.xilesou.top/pdf/1902.06327)

- 2019, **mobicom**, [Occlumency: Privacy-preserving Remote Deep-learning Inference Using SGX](https://gfsoso.99lb.net/sci-hub.html)

- 2020, **arXiv**, [CrypTFlow: Secure TensorFlow Inference](https://arxiv.org/pdf/1909.07814)

- 2020, **S&P**, [Secure Evaluation of Quantized Neural Networks]()

## Survey

- 2017, **arXiv**, [A Survey of Model Compression and Acceleration for Deep Neural Networks](https://arxiv.xilesou.top/pdf/1710.09282)

- 2018, **arXiv**, [A Survey of Machine and Deep Learning Methods for Internet of Things (IoT) Security](https://arxiv.xilesou.top/pdf/1807.11023)

- 2018, **ECCV**, [AI Benchmark: Running Deep Neural Networks on Android Smartphones](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Ignatov_AI_Benchmark_Running_Deep_Neural_Networks_on_Android_Smartphones_ECCVW_2018_paper.pdf)

- 2019, **arXiv**, [A Survey on Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection](https://arxiv.xilesou.top/pdf/1907.09693)

- 2019, **arXiv**, [Adversarial Attacks and Defenses in Images, Graphs and Text: A Review](https://arxiv.org/pdf/1909.08072)

- 2019, **arXiv**, [AI Benchmark: All About Deep Learning on Smartphones in 2019](https://arxiv.xilesou.top/pdf/1910.06663)

- 2019, **arXiv**, [Edge Intelligence: Paving the Last Mile of Artificial Intelligence with Edge Computing](https://arxiv.xilesou.top/pdf/1905.10083)

- 2019, **ICLR**, [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.xilesou.top/pdf/1903.12261)

- 2019, **TNN&LS**, [Adversarial Examples: Attacks and Defenses for Deep Learning](https://arxiv.xilesou.top/pdf/1712.07107)

- 2020, **arXiv**, [A Review on Generative Adversarial Networks: Algorithms, Theory, and Applications](https://arxiv.org/pdf/2001.06937)

## Other links
> [Paper List of Adversarial Examples](https://nicholas.carlini.com/writing/2019/all-adversarial-example-papers.html)
>
> [Paper List of Network Pruning](https://github.com/he-y/Awesome-Pruning)
>
> [Paper List of NLP Adversarial Examples](https://github.com/thunlp/TAADpapers)
