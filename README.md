

---

# README

## 1. Project Overview
The GA2AD (Generalized Adaptive Adversarial traning framework for Autonomous Driving) is designed to enhance the robustness and safety of autonomous vehicles (AVs) in mixed traffic environments by creating adaptive risky scenarios through the adjustment of adversarial maneuvers of Background Vehicles (BVs). This framework combines reinforcement learning (RL)-based and rule-based components, incorporating hybrid risk field-based inverse risk assessment and activation function modules to dynamically adjust the intensity of adversarial actions. The GAABV framework is capable of reducing collision rates by 55-70% and improving average absolute jerk by up to 35%, while maintaining operational efficiency with minimal reductions in average speed.

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
1. **Install Python**: Ensure you have Python 3.8 or higher installed. Download it from [Python's official website](https://www.python.org/downloads/).
2. **Install Dependencies**: Use pip to install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. **Install SUMO**: Follow the installation instructions on the [SUMO website](https://sumo.dlr.de/docs/Installing/index.html).
4. **Clone the Repository**: Clone this repository to your local machine:
   ```bash
   git clone https://github.com/MoMingQimio/GA2AD.git
   cd GAABV
   ```
5. **Run Main Script**: Start training by running:
   ```bash
   python main.py
   ```

### 3.2 Typical Installation Time
The installation process typically takes **10-15 minutes** on a standard desktop computer, depending on your internet connection and system performance.

## 4. Demo
To run the demo, simply execute:
```bash
python main.py
```
The demo will generate an `output.txt` file in the `results/` directory, containing metrics such as collision rate, average speed, and average absolute jerk.

## 5. Instructions for Use
1. Modify the input parameters in `config.py` to match your dataset.
2. Run the main script:
   ```bash
   python main.py
   ```
3. Results will be saved in the `results/` directory.

## 6. License Information
This software is licensed under the **MIT License**. Users are free to use, copy, modify, and distribute the software for any purpose, provided that the license notice is retained. For the full license text, please see [MIT License](https://opensource.org/licenses/MIT).

## 7. Issues and Support
If you encounter any problems or have questions about the software, please contact us via the following methods:
- Submit an Issue on the GitHub repository.
- Send an email to ciyusheng@hit.edu.cn or ciliang.lc@gmail.com..

## 8. Acknowledgements
This research was funded by the National Key R & D Program of China (2023YFB2603500), National Natural Science Foundation of China (No. 52402493), and Heilongjiang Provincial Natural Science Foundation of China (No.LH2023E055 and No. LH2024E059).

