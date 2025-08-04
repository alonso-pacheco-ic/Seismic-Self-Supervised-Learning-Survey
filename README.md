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
  
#### Based on Local redundancy and structural continuity
- **Neighbor2Neighbor**  
  Predicts a central trace using randomly sampled neighbors, encouraging the model to learn structural continuity while preserving geological features through regularized loss functions.  
  → [[paper]](https://doi.org/10.1190/geo2023-0895.1)
- **Noise2Clean**  
  Combines neighborhood sampling with expectation regularization to reconstruct cleaner traces without requiring clean references, enhancing robustness to real seismic noise.  
  → [[paper]](https://doi.org/10.1190/geo2023-0772.1)
- **Noiser2Noiser**  
  Applies re-corruption and symmetric loss to train models directly on noisy data, leveraging self-supervision without clean targets. Often implemented using variants of U-Net, DnCNN, or siamese CNNs.  
  → [[paper]](https://doi.org/10.1190/geo2023-0746.1)

#### Blind-Trace-Based Techniques

- **Blind-Trace Deblending**  
  Uses a U-Net variant trained without access to the target trace, forcing reconstruction from neighbors only. Includes rotation for vertical context and a second refinement stage called amplitude tuning.  
  → [[paper]](https://doi.org/10.1190/geo2022-0269.1)
- **Multi-Blind-Trace Learning**  
  Extends the blind-trace approach using multiple exclusion masks and a hybrid loss to better suppress signal leakage during training.  
  → [[paper]](https://doi.org/10.1190/geo2023-0305.1)
- **Semi-Blind-Trace Learning**  
  Uses adaptive masking and a refined loss to preserve vertical coherence while still excluding target traces during prediction.  
  → [[paper]](https://doi.org/10.1190/geo2023-0582.1)
- **Noisy-as-Clean Strategy**  
  Reconstructs clean signals from noisy inputs without ground truth by treating noisy traces as supervision, promoting spatial generalization.  
  → [[paper]](https://doi.org/10.1190/geo2023-0620.1)
- **Blind-Trace Network (BTN)**  
  Reconstructs missing data from decimated seismic records using spectral suppression and mixed training, while avoiding aliasing.  
  → [[paper]](https://doi.org/10.1190/geo2022-0051.1)
- **Pseudo-label Generation via Masking (PGM)**  
  Fills in missing traces using pseudo-labels generated through random trace masking. Trained with UNet++ and hybrid loss for enhanced reconstruction.  
  → [[paper]](https://doi.org/10.1109/TGRS.2022.3193986)


  
---

### 🧲 Contrastive Learning Techniques
- **Barlow Twins**  
  → [Author et al., Year](link_to_paper)

- **SimCLR / MoCo**  
  → [Author et al., Year](link_to_paper)

- **Triplet Loss Variants**  
  → [Author et al., Year](link_to_paper)

---

### 🧠 Masked and Generative Pretext Tasks
- **Masked Autoencoders (MAE)**  
  → [Author et al., Year](link_to_paper)

- **SimMIM / BEiT-style approaches**  
  → [Author et al., Year](link_to_paper)

- **Variational Autoencoders / GANs**  
  → [Author et al., Year](link_to_paper)

---

## Citation

If you find this work helpful, please cite the original survey (add citation here once published or add DOI/preprint link).

---

## Contact

For questions or collaboration inquiries, feel free to contact:  
**Alonso Pacheco**  
Email: [your_email@domain.com]

---

## License

This repository is licensed under the MIT License.
