# Rural Health Triage Agent

## Project Overview
This project is a command-line intelligent agent designed to act as a preliminary triage tool for low-resource environments. It utilizes Supervised Learning (Classification) concepts from the Fundamentals of AI and ML course to predict the likelihood of common diseases based on a patient's reported symptoms. 

## Prerequisites
* Python 3.8+
* pip (Python package installer)

## Environment Setup and Dependency Installation
Follow these exact steps to set up the environment:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-github-username>/<repo-name>.git
   cd <repo-name>

2. **Create and activate a virtual environment:**
   Windows:
   ```DOS
   python -m venv venv
   venv\Scripts\activate
   ```

   *macOS/Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

## Configuration & Training

This repository includes a pre-trained machine learning model (`triage_model.pkl` and `model_features.pkl`). **You do not need to train the model to use the application.** **[Optional] Retraining the Model:**
If you wish to rebuild the model from scratch, you can run the training script from the project root:

`python train_model.py`

This script will read the raw medical data from `symptoms.csv`, process the categorical features using One-Hot Encoding, train a new Random Forest Classifier, and overwrite the existing model files.

## Execution
This project is fully executable via the command line.

To launch the interactive triage agent, execute:
python triage_agent.py

## Usage:

* The terminal will present a series of yes/no questions regarding specific symptoms (e.g., "Are you experiencing a severe headache? (y/n)").

* Input y for yes or n for no, then press Enter.

* Once all questions are answered, the script will process the inputs through the trained model and output the top predicted conditions along with their probability scores directly in the terminal.



