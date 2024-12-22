from data_processing import load_and_process_data
from gpt_interface import execute_calc_with_manifestos
from visualization import create_comparison_plot
import numpy as np
import json

def main():
    # Set the cutoffs for parties and questions
    json_file_path = "Party_Answers_Converted_de.json"
    
    # Step 1: Generate AI answers
    print("Generating AI answers...")
    ai_matrix = execute_calc_with_manifestos(json_file_path)
    print("AI answers generated and saved to 'results.json' and 'results.csv'")
    
    # Step 2: Load and process both original and AI data
    print("\nProcessing data for comparison...")
    original_matrix, questions, party_names = load_and_process_data(json_file_path)
    
    # Load short questions from JSON file
    with open('short_question.json', 'r', encoding='utf-8') as file:
        short_questions = json.load(file)['questions']
    
    # Replace long questions with short versions
    questions = short_questions[:len(questions)]

    # Step 3: Create and save comparison plot
    print("\nCreating comparison plot...")
    create_comparison_plot(original_matrix, ai_matrix, questions, party_names)
    print("Comparison plot saved as 'comparison_plot.png'")


    # Print statistics
    agreement = np.sum(original_matrix == ai_matrix)
    total = original_matrix.size
    print(f"\nStatistics:")
    print(f"Total number of answers: {total}")
    print(f"Number of matching answers: {agreement}")
    print(f"Agreement percentage: {(agreement/total)*100:.2f}%")

if __name__ == "__main__":
    main()