# Artifacts for "Robust Intrusion Detection Systems using Quantum-Inspired Density Matrix"

This repository contains the source code and instructions to reproduce the experimental results presented in our paper, "Robust Intrusion Detection Systems using Quantum-Inspired Density Matrix".

## 📝 Description

The core contribution of our work is a novel network payload encoding method, the Density Matrix (DM) encoding. This repository provides the implementation of our method and the scripts required to reproduce the key experiments from the paper

## 🚀 Evaluation Workflow

The project is organized into several Jupyter notebooks and Python scripts. While you can run the entire pipeline from scratch, we provide pre-computed results and pre-trained models to make verification easier.

The general workflow is divided into three main stages:

---

### **Stage 1: Model Training**

This stage covers hyperparameter discovery and the training of all models used in the paper. You can skip this stage and use our pre-trained models if you only wish to generate the final results.

1. `model_discovery.ipynb`: Finds the optimal hyperparameters for the models on each dataset. This is optional, as the final hyperparameters are already saved in the `results/model_discovery` directory.
2. `defenses_training.ipynb`: Trains the baseline models and the adversarially trained models (PGD, MART, TRADES).

---

### **Stage 2: Attack Evaluation**

This notebook performs the white-box attack evaluations against the trained models.

3. `whitebox_attacks.ipynb`: Runs the FGSM, C&W, and JSMA attacks. This step can be time-consuming and is optional, as all attack results have been pre-computed and are available in the `results/attack_sweep` directory.

---

### **Stage 3: Generating Results**

This is the final step, which uses the pre-computed data from the previous stages to generate the plots and tables presented in the paper.

4. `results.ipynb`: This is the main notebook for verification. Running it will reproduce the key figures and tables that support our claims.

## Claims

Our artifacts support the following major claims from the paper:

* <b>(C1)</b>: <i>The DM encoding is empirically more stable (exhibits a lower Lipschitz constant) than the Stats method. (Supports Figure 2)</i>

   Run the script: `python lipschitz_evaluation.py` to generate Figure 2, which can be found precomputed in `results/lipschitz_ratios/Plots`.

* <b>(C2)</b>: <i>Under baseline white-box attacks, the DM encoding provides superior adversarial robustness compared to Stats and Raw encodings. (Supports Table 3)</i>

    Supporting results can be found as outputs from `results.ipynb` under the `BASELINE` heading, which are already precomputed. 

* <b>(C3)</b>: <i>Defensive training can boost the robustness of the DM encoding against strong attacks like C&W. (Supports Table 4)</i>

    Supporting results can be found as outputs from `results.ipynb` under the `DEFENSES` heading, which are already precomputed. 

* <b>(C4)</b>: <i>Generating attacks against the high-dimensional DM encoding is computationally more expensive, acting as a practical deterrent. (Supports Table 6)</i>

    The wall-clock time required to generate each attack is printed to the terminal as the `whitebox_attacks.ipynb` script runs, providing the raw data for Table 6.


---

## ⚙️ Setup & Installation

To get started, clone the repository and install the required dependencies. We recommend using a virtual environment.

```bash
# Clone the repository
git clone https://github.com/QIMLResearch/DensityMatrixRobustness.git
cd DensityMatrixRobustness

# Create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt