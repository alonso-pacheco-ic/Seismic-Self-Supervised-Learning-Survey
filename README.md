# Seismic Self-Supervised Learning Survey

This repository provides a structured summary of self-supervised learning (SSL) techniques applied to seismic data, based on our literature review:  
**"A Comprehensive Survey on Self-Supervised Learning Techniques in Seismic Data Analysis."**

## Overview

Seismic interpretation plays a critical role in subsurface analysis for oil and gas exploration. In recent years, machine learning has increasingly automated tasks such as facies segmentation, fault detection, and seismic data enhancement. However, the need for large amounts of labeled data poses a significant challenge, as labeling is time-consuming, expensive, and relies on domain expertise.

Self-Supervised Learning (SSL) has emerged as a promising approach to reduce dependency on labeled data by leveraging pretext tasks that learn useful representations from raw seismic volumes. Despite its potential, the application of SSL in geophysics remains scattered, with no clear consensus on the most effective strategies.

This survey aims to:
- Provide a systematic review of SSL methods applied to seismic data.
- Categorize these techniques by paradigm (e.g., context-based, contrastive, generative).
- Highlight their strengths, limitations, and typical use cases.
- Suggest future research directions for improved SSL adoption in seismic workflows.

---

## SSL Techniques

Below is a categorization of the main SSL methods found in the reviewed literature:

### Context-Based Techniques

#### Spatial Tasks
- **Rotation Prediction**  
  Learn to recognize the correct rotation angle applied to seismic patches, encouraging spatial awareness in feature representations.  
  → [[paper]](https://doi.org/10.1109/LGRS.2022.3193567)
- **Jigsaw Puzzle Solving**  
  Reconstruct the original spatial arrangement of shuffled seismic patches, promoting learning of spatial relationships and structural patterns.  
  → [[paper]](https://doi.org/10.1109/LGRS.2022.3193567)
- **Temporal Order Prediction**  
  Predict the correct temporal sequence of shuffled seismic trace segments to model temporal continuity and waveform dependencies.  
  → [[paper]](https://doi.org/10.1109/LGRS.2022.3193567)

#### Structural Tasks
- **Seismic-Domain Auxiliary Tasks**  
  Employ auxiliary tasks (e.g., waveform classification or channel tracking) as regularization strategies to improve generalization in seismic segmentation tasks.
  → [[paper]](https://doi.org/10.1109/LGRS.2023.3328837)
  
#### Focus on Denoising
- **Neighbor2Neighbor**  
  Predicts a central trace using randomly sampled neighbors, encouraging the model to learn structural continuity while preserving geological features through regularized loss functions.  
  → [[paper]](https://doi.org/10.1007/s11004-023-10089-3)
- **Noise2Clean**
  Combines neighborhood sampling with expectation regularization to reconstruct cleaner traces without requiring clean references, enhancing robustness to real seismic noise.  
  → [[paper]](https://doi.org/10.1007/s11600-023-01105-5)
- **Noiser2Noiser**  
  Applies re-corruption and symmetric loss to train models directly on noisy data, leveraging self-supervision without clean targets. Often implemented using variants of U-Net, DnCNN, or siamese CNNs.  
  → [[paper]](https://doi.org/10.1190/geo2023-0762.1)
- **Noise2Noise**  
  Uses pairs of noisy blocks from the same region, assuming uncorrelated noise and shared signal for denoising.  
  → [[paper]](https://doi.org/10.1109/LGRS.2022.3145835)
- **Tied Autoencoder**  
  Enforces symmetry between encoder and decoder to extract consistent features from noisy input.  
  → [[paper]](https://doi.org/10.3390/min11101089)
- **DNLR (Deep Nonlocal Regularizer)**  
  Applies nonlocal regularization on similar patches (via block matching) and integrates it into a U-Net-based CNN.
  → [[paper]](https://doi.org/10.1190/xu2023deep) [[code]](https://github.com/XuZitai/DNLR)

#### Blind-Trace-Based Techniques
- **Blind-Trace Deblending**  
  Uses a U-Net variant trained without access to the target trace, forcing reconstruction from neighbors only. Includes rotation for vertical context and a second refinement stage called amplitude tuning.  
  → [[paper]](https://doi.org/10.1109/TNNLS.2022.3188915)
- **Multi-Blind-Trace Learning**  
  Extends the blind-trace approach using multiple exclusion masks and a hybrid loss to better suppress signal leakage during training.  
  → [[paper]](https://doi.org/10.3997/2214-4609.202310269)
- **Semi-Blind-Trace Learning**  
  Uses adaptive masking and a refined loss to preserve vertical coherence while still excluding target traces during prediction.  
  → [[paper]](http://dx.doi.org/10.1111/1365-2478.13448) [[code]](https: //github.com/mahdiabedi/semi-blind-trace-deep-learning)
- **Noisy-as-Clean Strategy**  
  Reconstructs clean signals from noisy inputs without ground truth by treating noisy traces as supervision, promoting spatial generalization.  
  → [[paper]](https://doi.org/10.1109/TGRS.2024.3497163)
- **Blind-Trace Network (BTN)**  
  Reconstructs missing data from decimated seismic records using spectral suppression and mixed training, while avoiding aliasing.  
  → [[paper]](https://doi.org/10.1109/TGRS.2022.3167546)
- **Pseudo-label Generation via Masking (PGM)**  
  Fills in missing traces using pseudo-labels generated through random trace masking. Trained with UNet++ and hybrid loss for enhanced reconstruction.  
  → [[paper]](https://doi.org/10.1109/TGRS.2022.3148994)

#### Blind-Spot-Based Techniques
- **Noise2Void (Blind Spot Network)**  
  Masks central pixels during training to predict them from surrounding pixels, assuming noise is independent of signal.  
  → [[paper]](https://doi.org/10.1109/CVPR.2019.00223)
- **Blind Spot Visualization (BSV)**  
  Dual-branch network: one performs BSN-style denoising, the other reconstructs masked regions via a Blind Spot Mapper. Trains directly on noisy data and improves DAS denoising.  
  → [[paper]](https://doi.org/10.1109/LGRS.2024.3400836)
- **StructBS + Plug-and-Play ADMM**  
  Combines StructBS blind-spot network with iterative optimization for pseudo-deblended gathers. Uses temporally masked U-Net.  
  → [[paper]](https://arxiv.org/abs/2205.15395)
- **J-Invariant Masking (Noise2Self)**  
  Ignores values at masked locations while reconstructing them, encouraging learning from spatial context. Applied with modified U2Net for seismic data.  
  → [[paper]](https://doi.org/10.1190/geo2023-0640.1)
- **Amplitude-Preserving Blind Spots**  
  Modified U2Net with dilated convolutions and no batch norm or sigmoid layers, ensuring amplitude preservation.  
  → [[paper]](https://doi.org/10.1109/TGRS.2023.3307424)
- **SDeNet (Seismic DAS Denoising)**  
  Introduces context-aware blind spots with dilated convolutions and asymmetric downsampling to handle spatially correlated noise.  
  → [[paper]](https://doi.org/10.1190/geo2022-0641.1)
- **Autoencoder with Tied Weights**  
  Prevents input copying via encoder–decoder weight tying, enforcing implicit blind spots in reconstruction.  
  → [[paper]](https://doi.org/10.3390/min11101089)
- **Noise2Void Adapted to Seismic**  
  Masks noisy pixels and reconstructs them from context, adapting original BSN principles to seismic characteristics.  
  → [[paper]](https://www.sciencedirect.com/science/article/pii/S2666544121000277)

#### Custom Masking
- **Custom Structured Masking**  
  Selectively masks regions to preserve geological details and suppress coherent noise, using spatial context for reconstruction.  
  → [[paper]](https://doi.org/10.1109/IGARSS52108.2023.10283058), [[paper]](https://arxiv.org/abs/2310.13967)

#### Bernoulli-Based 
- **Bernoulli Dropout Masking**  
  Randomly masks pixels or traces based on a Bernoulli distribution. Variants of U-Net with partial convolutions and self-ensembling reconstruct the signal from sparse observations.  
  → [[paper]](https://doi.org/10.1007/s11004-022-10032-y), [[paper]](https://doi.org/10.1109/LGRS.2022.3167999), [[paper]](https://doi.org/10.1109/ICSPCC55723.2022.9984626), [[paper]](https://doi.org/10.1109/TGRS.2023.3268554) [[code]](https://github.com/XuZitai/S2S-WTV)
- **Double Bernoulli Sampling + FFT**  
  Applies two-stage Bernoulli masking with gated convolutions and a frequency-based FFT module for reconstructing randomly missing 3D traces.  
  → [[paper]](https://doi.org/10.1109/TGRS.2024.3401130) [[code]](https://github.com/Ji-seismic/N2N_deblending)
- **3-DPCNN (Multitask Learning)**  
  Combines Bernoulli-masked partial convolutions with multitask 3D CNNs for joint denoising and reconstruction in complex seismic volumes.  
  → [[paper]](https://ui.adsabs.harvard.edu/link_gateway/2022ITGRS..6025923C/doi:10.1109/TGRS.2022.3225923) [[code]](https://github.com/caowei2020/self-supervised-3-DPCNN)

#### Focus on Interpolation  
- **Frequency-Aware Interpolation (U-Net)**  
  Recovers missing traces by decomposing inputs into low- and high-frequency components.  
  → [[paper]](https://doi.org/10.1109/TGRS.2023.3299284)
- **SSLI: Self-Supervised Learning for Interpolation**  
  Predicts randomly masked traces using contextual information from the convolutional receptive field.  
  → [[paper]](https://doi.org/10.1190/geo2022-0586.1)
- **Student–Teacher U-Net++ (Temporal Consistency)**  
  Applies exponential moving average (EMA) to stabilize predictions and enhance temporal coherence.  
  → [[paper]](https://doi.org/10.1007/s12145-024-01485-2)
- **Multi-Directional Reconstruction (3D Decoder)**  
  Fuses directional context using multiple encoder branches and a 3D decoder for trace interpolation.  
  → [[paper]](https://doi.org/10.3997/2214-4609.202310139)
- **Low-Frequency Recovery via Iterative U-Net**  
  Reconstructs low-frequency seismic content using a hybrid spectral loss and prior frequency cues.  
  → [[paper]](https://arxiv.org/abs/2401.07938)
  
---

### Contrastive Learning Techniques

- **Spatial Contrastive Learning on Seismic Volumes**  
  Leverages spatial proximity—rather than visual similarity—to form positive pairs across adjacent slices.
  → [[paper]](https://arxiv.org/abs/2206.08158) [[code]](https://github.com/olivesgatech/facies_classification_benchmark)
- **Salt3DNet using Barlow Twins**  
  Uses Barlow Twins for 3D seismic data, learning decorrelated but similar features from augmentations. Avoids negative pairs and improves salt dome segmentation. 
  → [[paper]](https://doi.org/10.1109/TGRS.2024.3394592)

---
  
### Masked Image Modeling

#### Classical MIM Techniques
- **MAE with Vision Transformer (ViT)**  
  Trained to reconstruct randomly masked patches (75%) from 2D seismic slices.
  → [[paper]](https://www.sciencedirect.com/science/article/pii/S0098300424002929) [[code]](https://github.com/lluizfernandotrindade/Natural_Gas_Segmentation), [[paper]](https://arxiv.org/abs/2309.02791)[[code]](https://github.com/shenghanlin/SeismicFoundationModel), [[paper]](https://pubs.geoscienceworld.org/seg/tle/article-abstract/44/2/96/651627/SeisBERT-A-pretrained-seismic-image-representation)
- **SimMIM with Swin Transformer**  
  Applies SimMIM to 3D seismic volumes, masking ~60% of input patches and reconstructing them directly from the encoder output. Achieves better performance than training from scratch or other SSL strategies.  
  → [[paper]](https://arxiv.org/abs/2310.17974)
  
#### Advanced Masked Reconstruction and Hybrid SSL Approaches
- **Transformer and CNN-based Masked Reconstruction**  
  SSL with masked reconstruction achieves performance close to full supervision using only 5–10% labeled data.  
  → [[paper]](https://doi.org/10.1190/geo2023-0508.1)
- **Few-shot Facies Classification (MT-ProtoNet)**  
  Combines masked trace reconstruction with prototype learning, enabling accurate facies classification with only 1–5 labeled sections.  
  → [[paper]](https://doi.org/10.1190/geo2022-0281.1)
- **Auxiliary Image Reconstruction in CNNs**  
  Adds image reconstruction as an auxiliary task in DeepLabV3+ (ResNet-18), improving segmentation with limited labels.  
  → [[paper]](https://doi.org/10.1109/ICIP40778.2020.9190798) [[code]](https://charlielehman.github.io/publication/s6/)
- **Masked Temporal Reconstruction in LSTMs**  
  Uses a bidirectional LSTM trained to reconstruct masked time sequences, demonstrating the benefit of SSL in temporal models.  
  → [[paper]](https://doi.org/10.1109/SIBGRAPI62404.2024.10716309)

#### Masked SSL for Denoising and Interpolation
- **SDT (Seismic Denoising Transformer)**  
  Combines Swin Transformers and CNNs using cutout-based masking to denoise complex seismic noise patterns.  
  → [[paper]](https://doi.org/10.1109/TGRS.2024.3368282)
- **IST (Irregular Spatial Transformer)**  
  A Swin Transformer model for trace interpolation with patch-based segmentation and adaptive masking strategies.  
  → [[paper]](http://dx.doi.org/10.1109/TGRS.2023.3317305)
- **DINN (Dip-Informed Neural Network)**  
  Enhances anti-aliasing interpolation with dip-guided deformable convolutions and low-pass filter initialization.  
  → [[paper]](https://doi.org/10.1109/TGRS.2024.3359247)
- **Multi-branch Masked Reconstruction**  
  Two U-Nets with different transformation strategies fused via convolution to recover weak and overlapping seismic signals.  
  → [[paper]](http://dx.doi.org/10.1109/TGRS.2024.3401832) [[code]](https://github.com/mahdiabedi)

### Classical Generative Modeling

- **SeisSegDiff (DDPM-based Generative Modeling)**  
  Applies Denoising Diffusion Probabilistic Models (DDPMs) to reconstruct seismic volumes from noise. The learned features feed an ensemble of MLPs for facies classification with minimal supervision.  
  → [[paper]](https://www.sciencedirect.com/science/article/pii/S0098300424003066) [[code]](https://github.com/tobi-ore/SeisSegDiff), [[paper]](https://www.sciencedirect.com/science/article/pii/S0098300424003121)
- **Latent Space Factorization (LSF)**  
  A self-supervised encoder-decoder model projects features into orthogonal subspaces to isolate structures (faults, horizons, salt domes) without labels. Avoids reverse diffusion; uses gradient-based sampling.  
  → [[paper]](https://arxiv.org/abs/2108.09605)[[code]](https://github.com/olivesgatech/Latent-Factorization)
- **Score-Based Generative Reconstruction**  
  Employs conditional score functions and Langevin dynamics to recover signals from noisy inputs without clean labels, enabling stochastic self-supervised learning.  
  → [[paper]](https://doi.org/10.1109/TGRS.2024.3421597) [[code]](https://github.com/mengchuangji/VI-Non-IID)
  
### Hybrid SSL Strategies

- **FaultCRL (Contrastive + Reconstruction Learning)**  
  A hybrid method combining contrastive learning with seismic-aware masked reconstruction for seismic faults.
  → [[paper]](https://doi.org/10.1016/j.eswa.2024.123617)
- **FaultCDR (Reconstruction + Disentangled Learning)**  
  A disentanglement-reconstruction approach that splits spatial and temporal features prior to reconstruction.
  → [[paper]](https://doi.org/10.1109/TGRS.2024.3512547)


## Citation

If you find this work helpful, please cite the original survey (add citation here once published or add DOI/preprint link).

---

## Contact

For questions or collaboration inquiries, feel free to contact:  
**Alonso Pacheco Huachaca**  
Email: [a291204@dac.unicamp.br]  
**Mauricio Cifuentes Ruiz**

---

## License

This repository is licensed under the MIT License.
