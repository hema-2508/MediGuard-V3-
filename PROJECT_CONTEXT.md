# MediGuard Project Context

## Goal

Build an AI system that analyzes prescriptions, predicts drug interactions, estimates patient risk, explains results and provides conversational medical assistance.

---

# Completed

## Model 1
Graph Neural Network Drug-Drug Interaction Prediction

Architecture
- GraphSAGE
- PyTorch Geometric
- RDKit
- Morgan Fingerprints
- Molecular Descriptors
- MLP Pair Classifier

Datasets
1. DrugBank DDI
2. BioSNAP DDI

Dataset columns

label
smile1
smile2

Pipeline

Drug CSV
↓

RDKit Features
↓

Drug Graph

↓

GraphSAGE

↓

Drug Embeddings

↓

Pair Classifier

↓

DDI Prediction

Features

- graph caching
- feature caching
- descriptor normalization
- early stopping
- checkpoint saving
- evaluation metrics
- prediction API

Current performance

ROC-AUC ≈ 0.987

PR-AUC ≈ 0.986

Accuracy ≈ 94.9%

Status

Model 1 completed.

---

## Model 2
NLP Medicine Extraction

Architecture
- BioBERT token classification
- spaCy preprocessing
- BIO tagging (MEDICINE, STRENGTH, DOSAGE, FREQUENCY, DURATION, ROUTE)
- FastAPI inference service

Input
Prescription text / OCR

Output

Medicine
Strength
Dosage
Frequency
Duration
Route
Confidence

Location
`model2/`

Status

Model 2 completed.

---

# Remaining Models

## Model 3

Brand Mapping

Brand Name

↓

Generic Medicine

↓

DrugBank SMILES

---

## Model 4

Personalized Risk Model

Uses

Age

Gender

Diseases

Lab values

Lifestyle

Adjusts interaction severity.

---

## Model 5

Symptom Reasoning

Predict possible adverse reactions from

Symptoms

Patient profile

Drug interactions

---

## Model 6

Explainability

Generate human-readable explanations using

Attention

SHAP

Rule templates

---

## Model 7

Conversational Orchestrator

Coordinates all previous models.

Workflow

Prescription
↓

Medicine Extraction

↓

Brand Mapping

↓

DDI Prediction

↓

Risk Prediction

↓

Symptom Reasoning

↓

Explanation

↓

Chat Response

---

Project Stack

Python

PyTorch

PyTorch Geometric

RDKit

Transformers

FastAPI

spaCy

Pandas

NumPy

Scikit-learn

SQLite/PostgreSQL

Docker

---

Coding Guidelines

- Modular architecture
- Production-ready
- No placeholder code
- Reusable utilities
- Type hints
- Logging
- Config driven
- Separate train/inference
- Clean README
- Scalable design