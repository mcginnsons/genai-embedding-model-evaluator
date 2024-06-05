import boto3
from dotenv import load_dotenv
import logging
import json
import os
import re

# Setting up a logger with default settings
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Load environment variables from the .env file into the environment
load_dotenv()
# Setting up the default boto3 session with a specified AWS profile name
boto3.setup_default_session(profile_name=os.getenv("profile_name"))

def evaluate_model_output(result, knowledge_base, contexts):
    """
    Orchestrates the evaluation of model output across multiple evaluation criteria and calculates the final evaluation score.

    :param result: The golden output, extracted from the uploaded answer csv file.
    :param knowledge_base: The name of the knowledge base being evaluated.
    :param contexts: The extracted contexts from the knowledge base.
    :return: A tuple containing the final score and a summary of all the evaluation results.
    """
    # Construct a dictionary containing individual evaluation scores
    final_score_rubric = {
        "model_name": knowledge_base['name'],
        "faithfulness": result['faithfulness'],
        "answer_relevancy": result['answer_relevancy'],
        "context_precision": result['context_precision'],
        "context_recall": result['context_recall'],
        "context_entity_recall": result['context_entity_recall'],
        "answer_similarity": result['answer_similarity'],
        "answer_correctness": result['answer_correctness'],
        "harmfulness": result['harmfulness'],
        "maliciousness": result['maliciousness'],
        "coherence": result['coherence'],
        "correctness": result['correctness'],
        "conciseness": result['conciseness']

    }
    # Calculate the final score
    final_score = (result['faithfulness'] + result['answer_relevancy'] + result['context_precision'] + 
        result['context_recall'] + result['context_entity_recall'] + result['answer_similarity'] + result['answer_correctness']
        + result['coherence'] + result['correctness'] + result['conciseness']) / 10.0
    final_score -= ((result['maliciousness'] + result['harmfulness']) / 2.0)

    # Construct a summary of the evaluation results
    final_summary = f"""
Full Output:
{contexts}
---------------------------------------------------------------------

Knowledge Base Faithfulness: 
Score: {result['faithfulness']}
---------------------------------------------------------------------

Knowledge Base Answer Relevancy: 
Score: {result['answer_relevancy']}
---------------------------------------------------------------------

Knowledge Base Context Precision: 
Score: {result['context_precision']}
---------------------------------------------------------------------

Knowledge Base Context Recall: 
Score: {result['context_recall']}
---------------------------------------------------------------------

Knowledge Base Context Entity Recall: 
Score: {result['context_entity_recall']}
---------------------------------------------------------------------

Knowledge Base Answer Similarity: 
Score: {result['answer_similarity']}
---------------------------------------------------------------------

Knowledge Base Answer Correctness:
Score: {result['answer_correctness']}

---------------------------------------------------------------------

Knowledge Base Harmfulness:
Score: {result['harmfulness']}
---------------------------------------------------------------------

Knowledge Base Maliciousness:
Score: {result['maliciousness']}
---------------------------------------------------------------------

Knowledge Base Coherence:
Score: {result['coherence']}
---------------------------------------------------------------------

Knowledge Base Correctness:
Score: {result['correctness']}
---------------------------------------------------------------------

Knowledge Base Conciseness:
Score: {result['conciseness']}
------------------------------------------------------------------------------------------------------------------------------------------------------------
    """
    # Return the final score, final summary, and final scoring rubric
    return final_score, final_summary, final_score_rubric

def evaluate_model_performance(csv_string, model_id="anthropic.claude-3-sonnet-20240229-v1:0"):
    """
    Evaluates the performance of AI models based on provided CSV data.

    :param csv_string: A string containing CSV data with columns for 'Total Cost(1000)', 'Time Length', and 'Summary Score'.
    :param model_id: The ID of the model used for evaluation. Defaults to "anthropic.claude-3-sonnet-20240229-v1:0".
    :return: A string containing the analysis and findings of model performance based on the provided CSV data.
    """
    # Constructing a prompt for the Amazon Bedrock Claude 3 Sonnet model to analyze the provided CSV data
    prompt = f"""Human:

    Given the following CSV data on AI model performance:

    {csv_string}

    Please analyze the data and determine which model has the best performance in terms of cost efficiency and speed.
    
    'Total Cost(1000)' is the total cost in dollars for invoking the model 1000 times.
    'Time Length' is the time to invoke the model.
    'Score' is the invoke response quality score.

    Criteria for evaluation:
    1) The model with the lowest 'Total Embedding Cost(1000)' is considered as least expensive. 
    2) The model with the highest 'Total Embedding Cost(1000)' is considered as most expensive.
    3) The model with the shortest 'Time Length' is considered as fastest.
    4) The model with the highest 'Score' is considered as best answering result.
    4) The model with the lowest 'Score' is considered as worse answering result.

    Summarize your findings on which model performs best on each criterion and overall. Identify the percent time and cost difference.
    
    Format in markdown.
    """
    # Constructing the prompt object for model execution
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.5,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }
    boto3.setup_default_session(profile_name=os.getenv("profile_name"))
    # Creating a boto3 client for interacting with the Bedrock Runtime service
    client = boto3.client(
        service_name='bedrock-runtime',
        region_name=os.getenv("region_name")
    )
    # Invoking the Amazon Bedrock and the Claude 3 Sonnet model with the constructed prompt
    response = client.invoke_model(
        modelId=model_id, body=json.dumps(prompt)
    )
    # Extracting and parsing the response from the AI model
    response_body = json.loads(response.get('body').read())
    response = response_body['content'][0]['text']
    # Returns a string containing the analysis and findings of model performance based on the provided CSV data.
    return response
