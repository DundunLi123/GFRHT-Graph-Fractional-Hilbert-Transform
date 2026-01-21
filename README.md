# Graph Fractional Hilbert Transform: Theory and Application

This repository contains the source code and datasets for the paper:
**"Graph Fractional Hilbert Transform: Theory and Application"** *Authors: Daxiang Li and Zhichao Zhang* *Submitted to: MDPI (Under Review)*

## 📄 Abstract
The graph Hilbert transform (GHT) is a key tool in graph signal processing but is limited by fixed phase shifts and confinement to the spectral domain. This paper proposes the **Graph Fractional Hilbert Transform (GFRHT)**, a dual-parameter framework (fractional order $\alpha$ and angle $\beta$) that generalizes the GHT. The GFRHT enables analysis in arbitrary fractional domains and eliminates information loss for real-valued spectral components.

## 📂 Repository Structure

- `Code/`: Contains the implementation of GFRHT in MATLAB and Python.
  - `anomaly localization.m`: Scripts for Section 5.1.
  - `detection of diverse anomaly types across graph topologies.m`: Scripts for Section 5.2.
  - `anomaly_detection.py`: Scripts for Section 5.3.
  - `voice_classification.py`: Scripts for Section 5.4.
  - `edge_detection.m`: Scripts for Section 5.5.
- `Data/`:
  - `Molene/`: Preprocessed Molene Weather Dataset (Real-world sensor network).
  - `Synthetic/`: Generated graph signals for simulation.
  - `Voice/`: Voice signal classification data, Accessible in https://drive.google.com/drive/folders/1Eohh-4OtHFo8k3PiF3eDsrNcq3BkmXm8?usp=drive_link

## 🚀 Getting Started

### Prerequisites
* Python 3.12+
* PyTorch / NumPy / SciPy
* MATLAB R2024 

