from dotenv import load_dotenv
import boto3
import json
import os
from botocore.exceptions import ClientError
import logging
import streamlit as st
import csv

# Setting up a logger with default settings
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Loading environment variables from a .env file
load_dotenv()

if os.getenv("region_name") is None:
    region_name = 'us-east-1'
else:
    region_name = os.getenv("region_name")
    
# Setting up the default boto3 session with a specified AWS profile name
boto3.setup_default_session(profile_name=os.getenv("profile_name"))

# Instantiating the Amazon Bedrock Runtime Client

client = boto3.client(
    service_name="bedrock-agent-runtime",region_name=region_name)

# Define request headers, for Amazon Bedrock Model invocations
accept = 'application/json'
contentType = 'application/json'


def text_extraction(csv_path):
    """
    Extracts text from a CSV file.

    :param csv_path: The path to the csv file.
    :return: The extracted text from the csv file as a list of strings.
    """
    text = []

    try:
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                text.append(row[0])
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' does not exist.")
        return []
    except IndexError:
        print(f"Error: The CSV file '{csv_path}' does not have any data in the first column.")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []

    return text

def text_formatter(text_list):
    """
    Formats the extracted text into a string.

    :param text_list: The list of extracted text.
    :return: The formatted text as a string.
    """
    formatted_text = ""
    for i, text in enumerate(text_list):
        formatted_text +=  f"{i+1}. {text}\n"
    return formatted_text

def invoke_anthropic(model_id, prompt="", prompt_context="", max_tokens="4096"):
    """
    Invokes an Anthropic model using Amazon Bedrock and the specified parameters.

    :param model_id: The ID of the Anthropic model to invoke.
    :param prompt: Optional. The default prompt highlighting the task the model is trying to perform, defined in the orchestrator.py file.
    :param prompt_context: The prompt context includes the extracted text from the PDF file.
    :param max_tokens: Optional. The maximum number of tokens to generate. Defaults to the value of the 'max_tokens' environment variable.
    :return: A tuple containing the generated output text, the number of input tokens used, and the number of output tokens generated.
    """
    # Print the model ID (for debugging purposes)
    # TODO: Do we want to take this out?
    
    if max_tokens is None:
        max_tokens = "4096"
    
    # If prompt_context is provided, prepend it to the prompt
    if prompt_context:
        prompt=f"Human: \n\n {prompt} \n\n <context>{prompt_context}</context> \n Assistant: \n\n"
    # Define the request body for invoking the Anthropic model, using the messages API structure
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ],
    }

    try:
        # Invoke the Anthropic model through Bedrock using the defined request body
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
        )
        # Extract information from the response
        result = json.loads(response.get("body").read())
        # Extract the input tokens from the response
        input_tokens = result["usage"]["input_tokens"]
        # Extract the output tokens from the response
        output_tokens = result["usage"]["output_tokens"]
        # Extract the output text from the response
        output_text = result["content"][0]["text"]
        # Return the output text, input tokens, and output tokens
        return output_text, input_tokens, output_tokens
    except ClientError as err:
        # Log and raise an error if invoking the model fails
        logger.error(
            "Couldn't invoke {model_id}. Here's why: %s: %s",
            err.response["Error"]["Code"],
            err.response["Error"]["Message"],
        )
        raise