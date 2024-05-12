# Parkinson's Disease Classification

This repository contains a Python script for classifying Parkinson's disease using machine learning techniques. The script expands the dataset through bootstrapping, preprocesses it, and trains multiple classifiers to predict the presence of Parkinson's disease based on various features.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

Parkinson's disease is a neurodegenerative disorder that affects movement. Early detection and diagnosis are crucial for effective treatment. This script aims to aid in the classification of Parkinson's disease using machine learning algorithms. It expands the dataset by bootstrapping to increase its size and performs data preprocessing by scaling. Then, it trains and evaluates four classifiers:
- Support Vector Machine (SVM)
- Simple Neural Network
- Convolutional Neural Network (CNN)
- Random Forest

The accuracy of each model is reported on the test set, providing insights into their effectiveness in classifying Parkinson's disease.

## Dataset

The dataset used for this project is sourced from the UCI Machine Learning Repository. It contains biomedical voice measurements from 195 individuals, including 147 with Parkinson's disease. Each individual has multiple voice recordings, resulting in a dataset with various features related to speech.

You can find the dataset in root directory.

## Dependencies

The following Python libraries are required to run the script:
- Pandas
- Scikit-learn
- TensorFlow
- Keras
