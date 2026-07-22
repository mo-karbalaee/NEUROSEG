# NEUROSEG — Neuronal Soma Segmentation Pipeline

> Original project description (as provided by the project owner). Kept verbatim for reference.

## TL;DR

Develop a neuron segmentation pipeline using semantic segmentation for active neurons in microscopic Ca²⁺ imaging data (2D+t). Fluorescent calcium imaging helps understand brain functions by recording neuronal activities. However, manual annotation is extremely time-consuming and subjective. Therefore, an automatic tool should be developed.

## Background

Neurons communicate through electrical signals known as action potentials. When a neuron fires, intracellular calcium levels rise temporarily. Calcium imaging uses fluorescent indicators that emit stronger fluorescence when the calcium concentration increases. Active neurons "light up" over time when chronically imaged.

## Overall goal

The goal of this project is to develop a pipeline that automatically segments active neurons in calcium imaging data.

## Minimal viable product (Phase 0 deliverable)

The MVP should be a basic segmentation pipeline that runs end-to-end. The pipeline needs to include:

- loading the data,
- preprocessing,
- creating a basic semantic segmentation method, and
- plotting individual neural activity traces.

## Tasks

- Load and understand the calcium image dataset (2D+t)
- Preprocess data for training and inference
- Develop and compare unbiased and activity-based semantic segmentation methods
- Visualize and evaluate results

## Available data and code

Ca²⁺ imaging data from **mice, zebrafish and drosophila**.

Strategies:

- Unbiased via correlation: https://orgerlab.org/wp-content/uploads/2016/12/2016-Correlating.pdf
- Cellpose: https://github.com/MouseLand/cellpose

## Additional prerequisites

—

## Main contact

Sophie Hauser — sophie.louise.hauser@fau.de
