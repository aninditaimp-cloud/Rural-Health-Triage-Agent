# Rural Health Triage Agent

## Project Overview
This project is a command-line intelligent agent designed to act as a preliminary triage tool for low-resource environments. [cite_start]It utilizes Supervised Learning (Classification) concepts from the Fundamentals of AI and ML course [cite: 106] to predict the likelihood of common diseases based on a patient's reported symptoms. 

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
   python -m venv venv
   venv\Scripts\activate

   macOS/Linux:
   python3 -m venv venv
   source venv/bin/activate

3. **Install dependencies:**
   pip install -r requirements.txt

## Configuration
Before running the agent, you must train the classification model using the provided dataset.

Run the following command from the project root:
python train_model.py

This script reads the symptom data from data/symptoms.csv, trains the classifier, and generates a saved model file (triage_model.pkl) required for execution.

## Execution
This project is fully executable via the command line.

To launch the interactive triage agent, execute:
python triage_agent.py

## Usage:

* The terminal will present a series of yes/no questions regarding specific symptoms (e.g., "Are you experiencing a severe headache? (y/n)").

* Input y for yes or n for no, then press Enter.

* Once all questions are answered, the script will process the inputs through the trained model and output the top predicted conditions along with their probability scores directly in the terminal.



