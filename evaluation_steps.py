import boto3
from dotenv import load_dotenv
import logging
import json
import os
import asyncio
import aioboto3
import re

# Setting up a logger with default settings
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Load environment variables from the .env file into the environment
load_dotenv()
# Setting up the default boto3 session with a specified AWS profile name
boto3.setup_default_session(profile_name=os.getenv("profile_name"))

async def get_bedrock_client():
    """
    Asynchronously creates and returns a client for interacting with the Bedrock Runtime service.

    This function uses aioboto3, an asynchronous version of the AWS SDK for Python (Boto3), to create a client
    for the Bedrock Runtime service. It retrieves the AWS profile name and region name from environment variables.

    Returns:
        aioboto3.client: A client object for interacting with the Bedrock Runtime service.
    """
    
    if os.getenv("profile_name") is None:
        os.environ["profile_name"] = "default"
        os.environ["region_name"] = "us-east-1"
    
    # Create an aioboto3 session using the specified profile name
    session = aioboto3.Session(profile_name=os.getenv("profile_name"))
    # Asynchronously create a client for the Bedrock Runtime service
    async with session.client(
            service_name='bedrock-runtime',
            region_name=os.getenv("region_name"),

    ) as client:
        return client


async def model_execution(client, user_prompt, system_prompt):
    """
    Asynchronously executes a model using specified prompts, provided by each evaluation function
    and returns the score and evaluation summary.
    :param client: An aioboto3 client object for invoking Amazon Bedrock and the specific model.
    :param user_prompt: The user prompt used during model execution.
    :param system_prompt: The system prompt for the model execution and unique to the specific evaluation function.
    :return: A tuple containing the score of the evaluation and evaluation summary.
    """
    # Construct the content payload with user prompt
    content = [{
        "type": "text",
        "text": user_prompt
    }]
    # Construct the prompt object with model execution parameters, formatted for the Claue 3 Messages API
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10000,
        "temperature": 0,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }
    # Convert the prompt object to a JSON string
    prompt = json.dumps(prompt)
    # Invoke the model asynchronously with the provided prompt
    response = await client.invoke_model(
        body=prompt,
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        accept="application/json",
        contentType="application/json"
    )
    # Read the response body and parse it as JSON
    response_body = await response['body'].read()
    response_json = json.loads(response_body)
    # Extract the output text from the response
    output_text = response_json['content'][0]['text']
    # Extract score and evaluation summary from the output text
    score = parse_xml(output_text, "score").strip()
    evaluation_summary = parse_xml(output_text, "thoughts").strip()
    # Return the score and evaluation summary
    return score, evaluation_summary


def parse_xml(xml, tag):
    """
    Parse XML-like content to extract the value associated with a specific tag, handling one level of nested same tags.
    :param xml: The XML-like content as a string.
    :param tag: The tag whose value needs to be extracted.
    :return: The value associated with the specified tag or an empty string if the tag is not found.
    """
    try:
        # Construct a regex pattern to find content inside the specified tag,
        # This pattern attempts to skip over any nested tags of the same type.
        pattern = f'<{tag}>(?:<[^/]*?>.*?</[^>]*?>|[^<]*?)+</{tag}>'
        # Find the outermost tags first
        matches = re.findall(pattern, xml, re.DOTALL)
        if matches:
            # If there are matches, strip the outermost tag and find again to handle nested same tags
            clean_matches = []
            for match in matches:
                # Strip the outermost tag
                content = re.sub(f'^<{tag}>|</{tag}>$', '', match, flags=re.DOTALL)
                # Append cleaned content
                clean_matches.append(content)
            return " ".join(clean_matches).strip()
        return ""
    except re.error as e:
        # Return an empty string if a regex error occurs
        print(f"Regex error: {e}")
        return ""

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
