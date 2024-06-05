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