# HPI2-NN: Surrogate Model for Pellet Fuelling in Tokamak Discharges


Author: Alex Panera Alvarez


## 🧩 Overview

**HPI2-NN** is a machine learning surrogate model of the **HPI2 pellet ablation and deposition code** (https://gitlab.com/hpi2_group/HPI2_code), developed to accelerate integrated modeling of pellet-fuelled tokamak discharges.

The model learns from around 10000 **HPI2 simulations** on WEST experimental data and ITER simulation data, and predicts **pellet deposition profiles** based on plasma parameters and pellet injection conditions.  
It is designed for stand-alone and coupling into integrated modeling frameworks, enabling fast inference within integrated plasma scenario modeling.

---


## 🚀 Key Features

- Neural network surrogate trained on HPI2 synthetic data  
- PCA compression of temperature and density profiles  
- Supports ONNX inference for fast runtime execution  
- Modular code for evaluation  


---

## ⚙️ Installation

Clone the repository:
```bash
git clone https://github.com/DIFFER-NL/hpi2nn.git
cd hpi2nn
```
(Optional) Create a virtual environment and install dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
pip install onnxruntime
```
The onnxruntime version used for this project is 1.22.0

## Quick Inference Example

Shown on inference/simple_inference.py
Advised to be run from hpi2nn/ folder
```bash
python inference_hpi2nn/simple_inference.py
```

Inputs: Pellet radius in m, velocity in m/s, Te and Ti profiles in eV, ne profile in m-3, B0 in T, first point (R1,Z1) and second point (R2,Z2) in m
x coord preferred in rho_tor_norm, but using a_norm will not impact too much the result 
B0 is suppose to be negative always (inforced anyway in inference)

Outputs: deposition profile dne (m-3) and temperature change profile dTe (eV) same x coord as given in input

For JETTO implementation--> inference_hpi2nn/HPI2-NN_JETTO.py
FOR JETTO multiply ne by 1e6


## 📈 Training and Data

HPI2-NN was trained using synthetic data from HPI2 simulations under various plasma conditions representative of WEST and ITER configurations.
A different NN has been trained for every injection line and tokamak.


## 📘 Citation

A manuscript explaining and using this model is under preparation. So this repository is for the moment the only citable source.

## 👤 Author

Alex Panera Alvarez
PhD Candidate, Integrated Modelling Group — DIFFER
Email: a.paneraalvarez@differ.nl

GitHub: @alexpanera

## 📜 License

This project is licensed under the MIT License

.
© 2025 DIFFER — Dutch Institute for Fundamental Energy Research.

## 💡 Acknowledgements

This work was supported by EUROfusion under the Theory, Simulation, Verification and Validation (TSVV) tasks,
and carried out in collaboration with WEST and ITER Organization.

Special thanks to Florian Köchl, Eleonore Geulin for their contribution and the integrated modeling community for valuable discussions.
