import json
import numpy as np
import os
import re
import time
from dotenv import load_dotenv
import openai
import matplotlib.pyplot as plt
import seaborn as sns

# Disable parallelism for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Load environment variables from .env file
load_dotenv(dotenv_path='Backend/Evals/.env')

# Initialize OpenAI client with API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
openai_client = openai.OpenAI(api_key=api_key)

# Model specifications and cutoffs for questions and parties
modelspec = "gpt-4o" # gpt-4, gpt-4o, gpt-4o-mini
cutoff_questions = 0
cutoff_parties = 12

# Function to clean JSON strings by removing invalid control characters
def clean_json_string(json_string):
    json_string = re.sub(r'[\x00-\x1F\x7F]', '', json_string)
    return json_string

# Function to extract and return various specifications from the data file
def SpecsOfData(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_data = file.read()
        cleaned_data = clean_json_string(raw_data)
        data_Party = json.loads(cleaned_data)  # Load the cleaned JSON data
    
    # Get party names and full names from the root level
    party_names = data_Party['party_names']
    full_party_names = data_Party['party_full_names']
    party_names_length = len(party_names)

    # Get unique questions from party_answers
    unique_questions = set()
    for answer in data_Party['party_answers']:
        if answer['Party_Name'] == party_names[0]:
            unique_questions.add(answer['Question_Label'])

    # Convert set to list and apply cutoffs
    unique_questions = list(unique_questions)
    if cutoff_questions != 0:
        unique_questions = unique_questions[:cutoff_questions]
    
    if cutoff_parties != 0:
        full_party_names = full_party_names[:cutoff_parties]
        party_names = party_names[:cutoff_parties]

    num_unique_questions = len(unique_questions)
    party_names_length = len(party_names)

    return (
        "Germany",  # Hardcoded for this specific file
        party_names_length, 
        num_unique_questions, 
        data_Party,  
        party_names,
        full_party_names, 
        unique_questions, 
        data_Party['party_answers']
    )

# Function to create messages and behaviors for each question and party
def create_message(filepath):
    country,num_parties,num_questions,data, party_names,Party_Full_Names, questions, data_Country= SpecsOfData(filepath)


    
    messages_list = [["" for _ in range(len(Party_Full_Names))] for _ in range(len(questions))]
    behaviour_list = [["" for _ in range(len(Party_Full_Names))] for _ in range(len(questions))]

    for i in range(num_questions):
        for j in range(num_parties):
            messages_list[i][j] = f"question number {i}, question: {questions[i]}"
            behaviour_list[i][j] = (
                f'You are the political party {Party_Full_Names[j]} from {country}. '
                f'You will be asked a question that you have to answer in this JSON format: '
                f'"question" : "{questions[i]}", '
                f'"question_number" : "{i}", '
                f'"Full Party Name" : "{Party_Full_Names[j]}", '
                f'"AI_answer" : "<MUST BE EXACTLY ONE OF: disagree, neutral, agree>", '
                f'"AI_answer_reason" : "<your reasoning for your answer above, 2 sentences max.>", '
                f'"AI_confidence" : "<An integer number between 0 and 100 of the confidence of your answer>"'
            )

    return messages_list,behaviour_list

# Function to ask ChatGPT for an answer to a specific question for a specific party
def AskChatGPT(filepath, i, j, country):
    message2, behaviour2 = create_message(filepath)

    
    messages = [
        {"role": "system", "content": behaviour2[i][j]},
        {"role": "user", "content": message2[i][j]},
    ]
    temperature = 0
    max_tokens = 200 

    # Uncomment below for actual ChatGPT usage
    # response = openai_client.chat.completions.create(
    #     model=modelspec,
    #     messages=messages,
    #     temperature=temperature,
    #     max_tokens=max_tokens,
    #     top_p=1,
    #     frequency_penalty=1,
    #     presence_penalty=1
    # )
    # return response.choices[0].message.content
    return "neutral"

# Function to execute the calculation and generate the answer matrix and JSON using ChatGPT
def execute_calc2(filepath):
    country, num_party, num_questions, _, _, _, _, _ = SpecsOfData(filepath)
    GPT_Answer_Matrix = np.zeros((num_questions, num_party))
    k = 0
    start_time = time.time()
    answers_list = []

    for i in range(num_questions):
        for j in range(num_party):
            try:
                answer = AskChatGPT(filepath, i, j, country)
                answer_json = json.loads(answer)
                
                if isinstance(answer_json, dict):
                    # Convert word answer to number
                    answer_number = convert_answer_to_number(answer_json.get("AI_answer", "neutral"))
                else:
                    answer_number = 0
                    
                GPT_Answer_Matrix[i][j] = answer_number
                answers_list.append(answer_json)
                
            except Exception as e:
                print(f"Error processing response: {str(e)}")
                GPT_Answer_Matrix[i][j] = 0
                answers_list.append({
                    "question": f"Error processing question {i}",
                    "question_number": str(i),
                    "Full Party Name": "Error",
                    "AI_answer": "neutral",
                    "AI_answer_reason": f"Error: {str(e)}",
                    "AI_confidence": "0"
                })

            k += 1
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / k) * (num_questions * num_party)

            print(f"Progress {round(k / (num_party * num_questions) * 100, 2)}%, "
                  f"Question number {i + 1}/{num_questions}, Party number: {j + 1}/{num_party}: "
                  f"\n\n{answer}\n\n"
                  f"Elapsed time: {elapsed_time:.2f}s, Estimated total time: {estimated_total_time:.2f}s, "
                  f"Estimated remaining time: {estimated_total_time - elapsed_time:.2f}s ")


    csv_file = f"results.csv"
    np.savetxt(csv_file, GPT_Answer_Matrix, delimiter=",")

    return answers_list

# Add this function to convert the word answers to numbers
def convert_answer_to_number(answer):
    answer_map = {
        "disagree": -1,
        "neutral": 0,
        "agree": 1
    }
    return answer_map.get(answer.lower(), 0)  # returns 0 for any invalid input

def load_and_process_data(original_file, ai_results_file):
    # Load original data
    with open(original_file, 'r', encoding='utf-8') as file:
        raw_data = file.read()
        cleaned_data = clean_json_string(raw_data)
        original_data = json.loads(cleaned_data)
    
    # Load AI results
    with open(ai_results_file, 'r', encoding='utf-8') as file:
        ai_results = json.load(file)

    # Get party names and questions with cutoffs applied
    party_names = original_data['party_names'][:cutoff_parties]
    
    # Get unique questions from first party
    questions = []
    for answer in original_data['party_answers']:
        if answer['Party_Name'] == party_names[0]:
            questions.append(answer['Question_Label'])
    questions = questions[:cutoff_questions]  # Apply cutoff to questions

    # Create matrices for original and AI answers
    num_questions = len(questions)
    num_parties = len(party_names)
    
    original_matrix = np.zeros((num_questions, num_parties))
    ai_matrix = np.zeros((num_questions, num_parties))

    # Fill original matrix
    for answer in original_data['party_answers']:
        if answer['Party_Name'] in party_names:
            party_idx = party_names.index(answer['Party_Name'])
            if answer['Question_Label'] in questions:  # Only process questions within our cutoff
                q_idx = questions.index(answer['Question_Label'])
                original_matrix[q_idx][party_idx] = answer['Party_Answer']

    # Fill AI matrix
    k = 0
    for i in range(num_questions):
        for j in range(num_parties):
            try:
                ai_matrix[i][j] = convert_answer_to_number(ai_results[k]['AI_answer'])
            except (IndexError, KeyError) as e:
                print(f"Warning: Missing or invalid AI answer for question {i}, party {j}")
                ai_matrix[i][j] = 0  # Default to neutral for missing/invalid answers
            k += 1

    print(f"Processed {num_questions} questions for {num_parties} parties")
    return original_matrix, ai_matrix, questions, party_names

def create_comparison_plot(original_matrix, ai_matrix, questions, party_names):
    # Calculate difference matrix
    diff_matrix = np.abs(original_matrix - ai_matrix)
    
    # Create color matrix
    color_matrix = np.zeros_like(diff_matrix)
    color_matrix[diff_matrix == 0] = 2    # Full agreement (blue)
    color_matrix[diff_matrix == 1] = 1    # Partial agreement (light blue)
    color_matrix[diff_matrix == 2] = 0    # Disagreement (white)

    # Create plot
    plt.figure(figsize=(15, 10))
    colors = ['white', 'lightblue', 'blue']
    cmap = sns.color_palette(colors)
    
    sns.heatmap(color_matrix, 
                xticklabels=party_names, 
                yticklabels=questions,
                cmap=cmap,
                cbar=False)

    plt.title('Comparison between Original Party Answers and AI Predictions')
    plt.xlabel('Political Parties')
    plt.ylabel('Questions')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor='blue', label='Full Agreement'),
                      plt.Rectangle((0,0),1,1, facecolor='lightblue', label='Partial Agreement'),
                      plt.Rectangle((0,0),1,1, facecolor='white', label='Disagreement')]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig('comparison_plot.png', bbox_inches='tight', dpi=300)
    plt.show()

def create_original_matrix(json_file_path):
    # Load and clean JSON data
    with open(json_file_path, 'r', encoding='utf-8') as file:
        raw_data = file.read()
        cleaned_data = clean_json_string(raw_data)
        data = json.loads(cleaned_data)
    
    # Get party names and questions with cutoffs applied
    party_names = data['party_names'][:cutoff_parties]
    
    # Get unique questions from first party
    questions = []
    for answer in data['party_answers']:
        if answer['Party_Name'] == party_names[0]:
            questions.append(answer['Question_Label'])
    questions = questions[:cutoff_questions]
    
    # Create and fill the matrix
    num_questions = len(questions)
    num_parties = len(party_names)
    matrix = np.zeros((num_questions, num_parties))
    
    # Fill the matrix with answers
    for answer in data['party_answers']:
        if answer['Party_Name'] in party_names:
            party_idx = party_names.index(answer['Party_Name'])
            try:
                q_idx = questions.index(answer['Question_Label'])
                matrix[q_idx][party_idx] = answer['Party_Answer']
            except ValueError:
                continue  # Skip if question is not in our cutoff list
    
    # Save to CSV for verification
    csv_file = "original_matrix.csv"
    np.savetxt(csv_file, matrix, delimiter=",")
    
    return matrix, questions, party_names

# Main execution
if __name__ == "__main__":
    # Set the cutoffs for parties and questions
    cutoff_parties = 3  # Example: test with 3 parties
    cutoff_questions = 0  # Example: test with 3 questions
    
    json_file_path = "Party_Answers_Converted_de.json"
    
    # Step 1: Generate AI answers
    print("Generating AI answers...")
    ai_answers = execute_calc2(json_file_path)
    print("AI answers generated and saved to 'results.json' and 'results.csv'")
    
    # Step 2: Load and process both original and AI data
    print("\nProcessing data for comparison...")
    original_matrix, ai_matrix, questions, party_names = load_and_process_data(
        json_file_path,
        "results.json"
    )
    
    # Step 3: Create and save comparison plot
    print("\nCreating comparison plot...")
    create_comparison_plot(original_matrix, ai_matrix, questions, party_names)
    print("Comparison plot saved as 'comparison_plot.png'")
    
    # Print some statistics
    agreement = np.sum(original_matrix == ai_matrix)
    total = original_matrix.size
    print(f"\nStatistics:")
    print(f"Total number of answers: {total}")
    print(f"Number of matching answers: {agreement}")
    print(f"Agreement percentage: {(agreement/total)*100:.2f}%")


