import joblib
import pandas as pd
import sys
import time

MODEL_PATH = 'triage_model.pkl'

# The 10 common symptoms we will ask the user in the terminal
ASK_SYMPTOMS = [
    'high_fever', 'cough', 'headache', 'fatigue', 
    'chills', 'vomiting', 'nausea', 'joint_pain',
    'chest_pain', 'skin_rash'
]

def clear_screen():
    print("\n" * 40)

def print_header():
    print("======================================================")
    print("          ⚕️ RURAL HEALTH TRIAGE AGENT ⚕️          ")
    print("======================================================")
    print("Please answer the following questions with 'y' or 'n'.")
    print("Type 'exit' at any time to quit.\n")

def get_user_input(symptom):
    while True:
        display_name = symptom.replace('_', ' ').capitalize()
        response = input(f"Does the patient have {display_name}? (y/n): ").strip().lower()
        
        if response in ['y', 'yes']:
            return 1
        elif response in ['n', 'no']:
            return 0
        elif response == 'exit':
            print("\nExiting the triage agent. Stay healthy!")
            sys.exit()
        else:
            print("  -> Invalid input. Please type 'y' for yes or 'n' for no.")

def main():
    clear_screen()
    print("Loading medical database and predictive models...")
    
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"\n[ERROR]: Missing required model files.")
        print("Run 'python train_model.py' to generate them before starting the agent!")
        sys.exit(1)
        
    time.sleep(1)
    clear_screen()
    print_header()

    # 1. Ask user for the 10 core symptoms
    user_responses = {}
    for symptom in ASK_SYMPTOMS:
        user_responses[symptom] = get_user_input(symptom)

    print("\nAnalyzing symptoms through the classification model...")
    time.sleep(1.5)

    # 2. Build the exact input matrix the model expects using its internal memory
    expected_features = model.feature_names_in_
    
    # Initialize all 131 possible symptoms to [0]
    input_data = {col: [0] for col in expected_features}
    
    # Update only the features we asked about
    for symptom, val in user_responses.items():
        if symptom in expected_features:
            input_data[symptom] = [val]

    # Create the DataFrame and STRICTLY enforce the column order
    input_df = pd.DataFrame(input_data, columns=expected_features)

    # 3. Get predictions
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    results = list(zip(classes, probabilities))
    results.sort(key=lambda x: x[1], reverse=True)

    # 4. Display Results
    print("\n======================================================")
    print("                  DIAGNOSIS RESULTS                   ")
    print("======================================================")
    
    for i in range(min(3, len(results))):
        disease, prob = results[i]
        if prob > 0.01:
            print(f"{i+1}. {disease.upper()}: {prob * 100:.1f}% match")
            
    print("------------------------------------------------------")
    print("DISCLAIMER: This is a preliminary assessment tool and")
    print("does not replace a formal medical diagnosis.")
    print("======================================================\n")

if __name__ == "__main__":
    main()