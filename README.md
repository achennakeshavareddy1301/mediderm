# Multimodal AI Skin Disease Diagnosis System

**An End-to-End Deep Learning Framework for Automated Dermatological Diagnosis**



## Overview

This project presents a multimodal artificial intelligence system that integrates computer vision and natural language processing for automated skin disease diagnosis. The system combines a custom Vision Transformer (ViT) backbone with adaptive multi-scale attention, a BiomedicalBERT encoder for textual input, and a large language model (Mistral-7B) for generating structured clinical reports.

The framework provides a comprehensive diagnostic pipeline that interprets lesion images and associated patient symptom descriptions to produce an interpretable medical report, including differential diagnoses, severity estimation, and medication suggestions.

### Key Highlights

- Achieved 74.89% validation accuracy across 22 dermatological disease categories.  
- Trained on 41,914 clinically validated images from multiple dermatology datasets.  
- Utilizes a Multi-Point Observation (MPO) strategy to mitigate AI hallucinations.  
- Fully reproducible on free GPU environments (e.g., Google Colab T4).  
- Supports real-time inference with integrated clinical report generation.

---

## Features

### Clinical Features
- Automated classification of 22 distinct skin diseases.  
- Generation of differential diagnoses with confidence scores.  
- Personalized treatment and medication recommendations.  
- Follow-up and lifestyle guidance suggestions.  
- Severity assessment categorized as Mild, Moderate, or Severe.  

### Technical Features
- Custom Medical ViT with adaptive attention for dermatological imaging.  
- Multimodal alignment between image and text modalities using contrastive learning.  
- Integration with Mistral-7B LLM for contextual report generation.  
- Optimized inference pipeline for consumer-grade GPUs.  
- End-to-end trainable architecture supporting fine-tuning and transfer learning.

---

## System Architecture

The system consists of the following core components:

| Component | Architecture | Parameters | Purpose |
|------------|---------------|-------------|----------|
| Vision Backbone | Medical Adaptive Vision Transformer | ~22M | Extraction of visual lesion features |
| Text Encoder | Bio_ClinicalBERT | ~110M | Encoding of patient symptoms |
| Alignment Module | MLP Projector | ~1.5M | Alignment of visual and textual embeddings |
| LLM Projector | 3-layer MLP | ~4.7M | Projection to LLM embedding space |
| Report Generator | Mistral-7B-Instruct | 7B | Clinical report generation |

---

## Dataset

### Data Sources

| Dataset | Images | Source | Categories |
|----------|---------|---------|-------------|
| DermNet | 19,559 | Kaggle | 23 |
| Mgmitesh Skin Disease | 48,233 | Kaggle | 6 |
| Ismailpromus | 27,153 | Kaggle | 7 |
| **Total** | **94,945** | Combined | - |
| **Filtered** | **41,914** | Clinically relevant | 22 |

### Data Split
- Training: 80% (33,531 images)  
- Validation: 20% (8,383 images)  
- Stratified by class distribution  

---

## Model Performance

| Metric | Value |
|---------|--------|
| Validation Accuracy | 74.89% |
| Top-3 Accuracy | ~90% |
| Inference Time | 2–5 seconds per image |
| Model Size | ~150 MB |
| Training Time | 4 hours (ViT) + 2 hours (alignment modules) |

Hardware used: Google Colab T4 GPU (16 GB VRAM) and NVIDIA RTX 3070 (8 GB VRAM).

---

## Installation

### Prerequisites

- Python ≥ 3.8  
- CUDA ≥ 11.8 (for GPU support)  
- Minimum 16 GB RAM  



<img width="5167" height="5388" alt="sample_predictions" src="https://github.com/user-attachments/assets/1f2e514d-c897-4b8e-8f38-5ce4b34143c6" />
<img width="1207" height="991" alt="proj1-Page-3 drawio (2)" src="https://github.com/user-attachments/assets/2f2669c5-8bce-4109-b02d-0b36d46fe4ac" />

<img width="4759" height="3564" alt="training_history (1)" src="https://github.com/user-attachments/assets/e56ff53f-eb9d-4d28-b128-a5af843e2c11" />
<img width="3568" height="2672" alt="roc_curves" src="https://github.com/user-attachments/assets/a84c7ecc-fc23-42f9-a967-6afdf8fc050c" />



