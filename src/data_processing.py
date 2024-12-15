import json
import re
import numpy as np
from config import cutoff_questions, cutoff_parties

def clean_json_string(json_string):
    json_string = re.sub(r'[\x00-\x1F\x7F]', '', json_string)
    return json_string

def SpecsOfData(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_data = file.read()
        cleaned_data = clean_json_string(raw_data)
        data_Party = json.loads(cleaned_data)
    
    party_names = data_Party['party_names']
    full_party_names = data_Party['party_full_names']

    unique_questions = set()
    for answer in data_Party['party_answers']:
        if answer['Party_Name'] == party_names[0]:
            unique_questions.add(answer['Question_Label'])

    num_unique_questions = len(unique_questions)
    party_names_length = len(party_names)



    unique_questions = list(unique_questions)
    if cutoff_questions != 0:
        unique_questions = unique_questions[:cutoff_questions]
        num_unique_questions = cutoff_questions
    
    if cutoff_parties != 0:
        full_party_names = full_party_names[:cutoff_parties]
        party_names = party_names[:cutoff_parties]
        party_names_length = cutoff_parties


    return (
        "Germany",
        party_names_length,
        num_unique_questions,
        data_Party,
        party_names, 
        full_party_names,
        unique_questions,
        data_Party['party_answers']
    )

def convert_answer_to_number(answer):
    answer_map = {
        "disagree": -1,
        "neutral": 0,
        "agree": 1
    }
    return answer_map.get(answer.lower(), 0)

def load_and_process_data(original_file):
    with open(original_file, 'r', encoding='utf-8') as file:
        raw_data = file.read()
        cleaned_data = clean_json_string(raw_data)
        original_data = json.loads(cleaned_data)


    party_names = original_data['party_names']
    if cutoff_parties != 0:
        party_names = party_names[:cutoff_parties]
        num_parties = cutoff_parties
    
    questions = []
    for answer in original_data['party_answers']:
        questions.append(answer['Question_Label'])

    questions = list(set(questions))

    num_questions = len(questions)
    num_parties = len(party_names)
    
    if cutoff_questions != 0:
        questions = questions[:cutoff_questions]
        num_questions = cutoff_questions
    
    original_matrix = np.zeros((num_questions, num_parties))

    for answer in original_data['party_answers']:
        if answer['Party_Name'] in party_names:
            party_idx = party_names.index(answer['Party_Name'])
            if answer['Question_Label'] in questions:
                q_idx = questions.index(answer['Question_Label'])
                original_matrix[q_idx][party_idx] = answer['Party_Answer']

    print(f"Processed {num_questions} questions for {num_parties} parties")
    return original_matrix, questions, party_names

def create_original_matrix(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        raw_data = file.read()
        cleaned_data = clean_json_string(raw_data)
        data = json.loads(cleaned_data)
    
    party_names = data['party_names'][:cutoff_parties]
    
    questions = []
    for answer in data['party_answers']:
        if answer['Party_Name'] == party_names[0]:
            questions.append(answer['Question_Label'])
    questions = questions[:cutoff_questions]
    
    num_questions = len(questions)
    num_parties = len(party_names)
    matrix = np.zeros((num_questions, num_parties))
    
    for answer in data['party_answers']:
        if answer['Party_Name'] in party_names:
            party_idx = party_names.index(answer['Party_Name'])
            try:
                q_idx = questions.index(answer['Question_Label'])
                matrix[q_idx][party_idx] = answer['Party_Answer']
            except ValueError:
                continue

    csv_file = "original_matrix.csv"
    np.savetxt(csv_file, matrix, delimiter=",")
    
    return matrix, questions, party_names