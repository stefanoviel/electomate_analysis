import json
import time
import numpy as np

from datetime import datetime
from pathlib import Path
from config import openai_client, modelspec, cutoff_parties, cutoff_questions, is_rag_context
from data_processing import SpecsOfData, convert_answer_to_number

from llama_index.core import StorageContext, load_index_from_storage
import multiprocessing as mp
from functools import partial


# Load the index from storage
# print("Loading index...")
storage_context = StorageContext.from_defaults(persist_dir="index_store")
index = load_index_from_storage(storage_context)
# print("Index loaded!")



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



def AskChatGPT_with_context(filepath, i, j, country, index):
    message2, behaviour2 = create_message(filepath)
    
    # Extract the question from the prepared messages
    question = message2[j][i].replace("question: ", "")
    
    # Create a query engine with more comprehensive retrieval settings
    query_engine = index.as_query_engine(
        similarity_top_k=2,  # Retrieve top 5 most relevant chunks
        response_mode="tree_summarize"  # Synthesize information from multiple chunks
    )
    
    # Query with more detailed parameters
    llm_response = query_engine.query(
        question
    )
    
    # Get both the response and source nodes
    context = str(llm_response)
    source_nodes = llm_response.source_nodes
    
    # Build comprehensive context including source information
    detailed_context = context + "\n\nAdditional relevant information:\n"
    for idx, node in enumerate(source_nodes, 1):
        detailed_context += f"\nSource {idx}:\n{node.node.text}\n"
    
    # Create messages with enhanced context included
    messages = [
        {"role": "system", "content": behaviour2[j][i] + f"Base your answer primarily on the provided context from party documents.\nRelevant context from party documents:\n{detailed_context}\n\n Use this comprehensive context to inform your response."},
        {"role": "user", "content": message2[j][i]},
    ]
    
    print(messages)

    temperature = 0
    max_tokens = 200
    top_p = 0.1
    frequency_penalty = 0
    presence_penalty = 0

    response = openai_client.chat.completions.create(
        model=modelspec,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )

    response_content = response.choices[0].message.content

    country,num_parties,num_questions,data, party_names,Party_Full_Names, questions, data_Country= SpecsOfData(filepath)
    # Log the prompt and response

    # Remove ```json and ``` if present
    response_content = response_content.replace('```json', '').replace('```', '')
    try:
        return json.loads(response_content)
    except json.JSONDecodeError:
        print(f"Error decoding JSON response: {response_content}")
        return {}



# Function to ask ChatGPT for an answer to a specific question for a specific party
def AskChatGPT(filepath, i, j, country):
    message2, behaviour2 = create_message(filepath)

    messages = [
        {"role": "system", "content": behaviour2[j][i]},  # j is question index, i is party index
        {"role": "user", "content": message2[j][i]},
    ]
    temperature = 0
    max_tokens = 200 
    top_p = 0.1
    frequency_penalty = 0
    presence_penalty = 0

    # # Uncomment below for actual ChatGPT usage
    response = openai_client.chat.completions.create(
        model=modelspec,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    # print(response.choices[0].message.content)
    response_content = response.choices[0].message.content
    # Log the prompt and response

    country,num_parties,num_questions,data, party_names,Party_Full_Names, questions, data_Country= SpecsOfData(filepath)
    
    # Remove ```json and ``` if present
    response_content = response_content.replace('```json', '').replace('```', '')
    try:
        return json.loads(response_content)
    except json.JSONDecodeError:
        print(f"Error decoding JSON response: {response_content}")
        return {}

def process_question(args):
    i, j, filepath, country, is_rag_context, index = args
    try:
        if is_rag_context:
            response = AskChatGPT_with_context(filepath, i, j, country, index)
        else:
            response = AskChatGPT(filepath, i, j, country)
        time.sleep(0.1)  # Add a small delay to avoid hitting rate limits
        return (i, j, response)
    except Exception as e:
        print(f"Error processing question: {e}")
        return (i, j, {})

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
    # print("party_names_length", party_names_length)
    # print("num_unique_questions", num_unique_questions)
    
    # Create a matrix to store answers
    answer_matrix = np.zeros((num_unique_questions, party_names_length))
    
    # Prepare arguments for parallel processing
    args_list = [
        (i, j, filepath, country, is_rag_context, index)
        for i in range(party_names_length)
        for j in range(num_unique_questions)
    ]
    
    # Calculate total iterations for progress bar
    total_iterations = len(args_list)
    
    # Initialize the process pool
    num_processes = mp.cpu_count() - 1  # Leave one CPU core free
    pool = mp.Pool(processes=num_processes)
    
    # Process questions in parallel with progress tracking
    for idx, (i, j, response) in enumerate(pool.imap_unordered(process_question, args_list)):
        # Store the response in results list
        results.append({
            "Party_Name": party_names[i],
            "Question_Label": unique_questions[j],
            "Answer": response
        })
        
        # Convert response to numerical value (-1, 0, 1)
        try:
            # print('ai answer', response["AI_answer"])
            answer_matrix[j][i] = convert_answer_to_number(response["AI_answer"])
        except:
            print(f"Error converting answer to number: {response}")
            answer_matrix[j][i] = 0  # Default to neutral if there's an error
        
        # Update and display progress bar
        progress = int(50 * (idx + 1) / total_iterations)
        print(f"\rProgress: [{'=' * progress}{' ' * (50-progress)}] {idx + 1}/{total_iterations}", end='')
    
    # Close the pool
    pool.close()
    pool.join()
    
    print()  # New line after progress bar completes
    # Save the matrix to CSV
    np.savetxt("results_rag.csv", answer_matrix, delimiter=",")
    
    return answer_matrix
