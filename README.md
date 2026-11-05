# VisionGuard

[![VisionGuard CI](https://github.com/nihaal237/23I-0831_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/nihaal237/23I-0831_A2_Ops/actions/workflows/ci.yml)

AI vs Real Image Classification and Deployment

An end-to-end MLOps application that classifies uploaded images as
AI-generated or real using TensorFlow and Keras.

## Technologies
- TensorFlow / Keras
- DVC
- MLflow / DagsHub
- Flask
- PostgreSQL
- Docker / Docker Compose
- GitHub Actions

## Features
- AI-generated vs real image classification
- Experiment tracking and model versioning
- Web-based image upload and prediction
- Database-backed prediction history
- Cloud-hosted inference API

## CI/CD and Secrets Management

VisionGuard uses GitHub Actions to automate Python linting, unit tests, and Docker image builds on pushes and pull requests.

Sensitive credentials, including DagsHub tokens and deployment passwords, must be stored under **GitHub Repository Settings → Secrets and variables → Actions** rather than hardcoded into workflow YAML files.

The current CI workflow uses synthetic test assets and requires no DagsHub credentials. This prevents private tokens and trained model files from being exposed in the public repository.