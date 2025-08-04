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

### 🧩 Context-Based Techniques
- **Rotation Prediction** : identify the rotation angle of seismic patches
  → [paper](10.1109/LGRS.2022.3193567)

- **Jigsaw Puzzle Solving**  Network to recover the original arrangement of shuffled image patches
  → [paper](10.1109/LGRS.2022.3193567)

- **Temporal Order Prediction**  shuffled trace segments are reordered to recover event continuity
  → [paper](10.1109/LGRS.2022.3193567)

- **Seismic-Domain Auxiliary Tasks**:   as regularization to improve generalization in seismic channel segmentation
  → [paper](10.1109/LGRS.2023.3328837)

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
