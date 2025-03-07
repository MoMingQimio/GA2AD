

# README

## 1. Project Overview
This repository contains the implementation of a reinforcement learning-based environment for training autonomous vehicles (AVs) using the SUMO traffic simulator. The environment includes both a rule-based driver model and a reinforcement learning-based adversarial background vehicle (BV) model to simulate realistic traffic interactions and enhance the robustness of AVs in mixed traffic scenarios.

## 2. System Requirements

### 2.1 Software Dependencies
- Python: 3.8.20
- NumPy: 1.24.3
- SciPy: 1.10.1
- PyTorch: 2.4.1 (CPU version)
- Gym: 0.26.2
- Matplotlib: 3.7.2
- Pandas: 2.0.3
- OpenCV-python: 4.10.0.84
- SUMO: 1.21.0
- Jupyter: 7.2.2 

### 2.2 Operating Systems
- Windows 10

### 2.3 Hardware Requirements
- Modern CPU (Intel Core i5 or higher recommended)
- 16 GB RAM (32 GB recommended for large-scale simulations)
- NVIDIA GPU with CUDA support (optional but recommended for faster performance)

## 3. Installation Guide

### 3.1 Installation Steps
1. **Install Python**: Ensure you have Python 3.8 or higher installed. You can download it from [Python's official website](https://www.python.org/downloads/).
2. **Install Dependencies**: Use pip to install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. **Install SUMO**: Follow the installation instructions on the [SUMO website](https://sumo.dlr.de/docs/Installing/index.html) to set up the SUMO traffic simulator.
4. **Clone the Repository**: Clone this repository to your local machine:
   ```bash
   git clone https://github.com/MoMingQimio/GAABV.git
   cd GAABV
   ```
5. **Install the Package**: Run the following command to install the package:
   ```bash
   cd SUMO-RL-ENVIRONMENT
   pip install -e.
   ```

### 3.2 Typical Installation Time
The installation process typically takes **10-15 minutes** on a standard desktop computer, depending on your internet connection and system performance.

## 4. Demo

### 4.1 Dataset
A small simulated urban traffic dataset is included in the `data/` directory. This dataset is used to demonstrate the functionality of the environment.

### 4.2 Running the Demo
1. **Navigate to the Scripts Directory**:
   ```bash
   cd scripts
   ```
2. **Run the Demo Script**:
   ```bash
   python demo_script.py
   ```
3. **Expected Output**: The demo will generate a result file named `output.txt` in the `results/` directory. The output should include metrics such as collision rate, average speed, and average absolute jerk.
4. **Expected Runtime**: The demo typically takes **2-3 minutes** to run on a standard desktop computer.

## 5. Instructions for Use

### 5.1 Running on Your Data
1. Place your dataset in the `data/` directory.
2. Modify the input parameters in the `config.py` file to match your dataset.
3. Run the main script:
   ```bash
   python main_script.py
   ```
4. The results will be saved in the `results/` directory.

### 5.2 Reproducing Manuscript Results
To reproduce the results presented in the manuscript, follow these steps:
1. Download the full dataset from `[Dataset Link]`.
2. Place the dataset in the `data/` directory.
3. Run the following script:
   ```bash
   python reproduce_results.py
   ```
4. The results will be saved in the `results/` directory and should match the results presented in the manuscript.

## 6. License Information
This software is licensed under the **MIT License**. Users are free to use, copy, modify, and distribute the software for any purpose, provided that the license notice is retained. For the full license text, please see [MIT License](https://opensource.org/licenses/MIT).

## 7. Frequently Asked Questions

### 7.1 What should I do if I encounter issues during installation?
If you encounter issues during installation, please try the following:
- Ensure all dependencies are correctly installed.
- Verify that your Python version meets the requirements.
- If the problem persists, please submit an issue on the GitHub repository or contact the authors directly.

### 7.2 What should I do if the runtime is too long?
If the runtime is longer than expected, please try the following:
- Ensure your hardware meets the minimum requirements.
- Reduce the size of the dataset to test performance.
- If using a GPU, ensure that CUDA and the relevant drivers are correctly installed.

## 8. Issues and Support
If you encounter any problems or have questions about the software, please contact us via the following methods:
- Submit an [Issue](https://github.com/yourusername/yourrepository/issues) on the GitHub repository.
- Send an email to [your.email@example.com](mailto:your.email@example.com).

## 9. Acknowledgements
We would like to thank the following individuals and organizations for their contributions to this project:
- [Contributor Name]
- [Funding Agency]

---

### **Notes**
1. **Testing**: Before submission, ensure that at least one colleague who is unfamiliar with the software can install and run it according to the README instructions.
2. **Updates**: If the software or dependencies are updated, update the README accordingly.
3. **Clarity**: Ensure that the instructions are concise and clear to avoid confusion.

---

Feel free to customize this template to fit your specific needs!
