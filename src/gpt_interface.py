import json
import time
import numpy as np
from config import openai_client, modelspec, cutoff_parties, cutoff_questions
from data_processing import SpecsOfData, convert_answer_to_number


# Function to create messages and behaviors for each question and party
def create_message(filepath):
    country,num_parties,num_questions,data, party_names,Party_Full_Names, questions, data_Country= SpecsOfData(filepath)
    
    messages_list = [["" for _ in range(len(Party_Full_Names))] for _ in range(len(questions))]
    behaviour_list = [["" for _ in range(len(Party_Full_Names))] for _ in range(len(questions))]

    for i in range(num_questions):
        for j in range(num_parties):
            messages_list[i][j] = f"question: {questions[i]}"
            behaviour_list[i][j] = (
                f'You are the political party {Party_Full_Names[j]} from {country}. '
                f'You will be asked a question that you have to answer in this JSON format: '
                f'"question" : "{questions[i]}", '
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
        {"role": "system", "content": behaviour2[j][i]},  # j is question index, i is party index
        {"role": "user", "content": message2[j][i]},
    ]
    temperature = 0
    max_tokens = 200 

    # # Uncomment below for actual ChatGPT usage
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

    return "agree"

def execute_calc2(filepath):
    country, party_names_length, num_unique_questions, data_Party, party_names, full_party_names, unique_questions, party_answers = SpecsOfData(filepath)
    
    # Apply cutoffs
    if cutoff_parties > 0:
        party_names_length = min(cutoff_parties, party_names_length)
        party_names = party_names[:party_names_length]
        
    if cutoff_questions > 0:
        num_unique_questions = min(cutoff_questions, num_unique_questions)
        unique_questions = unique_questions[:num_unique_questions]
    
    results = []
    print("party_names_length", party_names_length)
    print("num_unique_questions", num_unique_questions)
    
    # Create a matrix to store answers
    answer_matrix = np.zeros((num_unique_questions, party_names_length))
    
    for i in range(party_names_length):
        for j in range(num_unique_questions):
            response = AskChatGPT(filepath, i, j, country)
            
            # Store the response in results list
            results.append({
                "Party_Name": party_names[i],
                "Question_Label": unique_questions[j],
                "Answer": response
            })
            
            # Convert response to numerical value (-1, 0, 1)
            try:
                answer_matrix[j][i] = convert_answer_to_number(response)
            except:
                answer_matrix[j][i] = 0  # Default to neutral if there's an error
    
    # Save the matrix to CSV
    np.savetxt("results.csv", answer_matrix, delimiter=",")
    
    return results
