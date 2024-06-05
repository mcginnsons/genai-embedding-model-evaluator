# GenAI Embedding Model Evaluator

The GenAI Embedding Model Evaluator is a tool designed for you to analyze and compare the performance of various Bedrock Embedding models, particularly focusing on aspects like faithfulness, answer relevancy, context recall, context precision, context entity recall, answer similarity, answer correctness, harmfulness, maliciousness, coherence, correctness, and conciseness as made available by the open source RAGAS framework. By automating the evaluation process and providing detailed scoring across multiple criteria, it enables you to make informed decisions when selecting the most optimal models for specific tasks. With its streamlite interface, you can easily upload your questions and answers, run evaluations on different embedding models and knowledge bases, and visualize comparative performance metrics to identify the best embedding model for your needs.

##### Authors: Justin McGinnity Forked from a branch created by: Brian Maguire, Dom Bavaro, Ryan Doty

## Demo
![Alt text](images/demo.gif)

## Features

- **Cost Efficiency and Speed Analysis:** Enables comparison of different embedding models based on their total operational costs and execution time to pinpoint the most efficient options.
- **RAGAS Evaluation with Detailed Scoring:** Leverages the open source RAGAS (RAG assessment) framework to evaluate the effectiveness of an embedding model and knowledge base to perform across the following metrics: faithfulness, answer relevancy, context recall, context precision, context entity recall, answer similarity, answer correctness, harmfulness, maliciousness, coherence, correctness, and conciseness. 
- **Visualization Tools for Model Comparison:** Provides visual aids to facilitate an easier understanding of how different models stack up against each other in performance metrics.
- **Automated Model Evaluation:** Streamlines the evaluation process through AI-driven analysis of model performance data.

## Getting Started

To begin using the GenAI Model Evaluator:

1. Ensure you have Amazon Bedrock Access and CLI Credentials.
2. Install Python 3.9 or 3.10 on your machine.
3. Clone this repository to your local environment.
4. Navigate to the project directory.
5. Optional: Set the .env settings
6. Set up a Python virtual environment and install required dependencies

### Configuration

Configure necessary environment variables (e.g., AWS credentials, database connections) as detailed in sample directories.

# How to use this Repo:

## Prerequisites:

1. Amazon Bedrock Access and CLI Credentials.
2. Ensure Python 3.10 installed on your machine, it is the most stable version of Python for the packages we will be using, it can be downloaded [here](https://www.python.org/downloads/release/python-3911/).

## Step 1:

The first step of utilizing this repo is performing a git clone of the repository.

```
git clone https://github.com/aws-samples/genai-embedding-model-evaluator.git

```


## Step 2:

Set up a python virtual environment in the root directory of the repository and ensure that you are using Python 3.10. This can be done by running the following commands:

```
pip install virtualenv
python3.10 -m venv venv

```

The virtual environment will be extremely useful when you begin installing the requirements. If you need more clarification on the creation of the virtual environment please refer to this [blog](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/).
After the virtual environment is created, ensure that it is activated, following the activation steps of the virtual environment tool you are using. Likely:

```
cd venv
cd bin
source activate
cd ../../

```

After your virtual environment has been created and activated, you can install all the requirements found in the requirements.txt file by running this command in the root of this repos directory in your terminal:

```
pip install -r requirements.txt

```
## Step 3:

Create a .env file in the root of this repo. Within the .env file you just created you will need to configure the .env to contain:

```
region_name=us-east-1
profile_name=<AWS_CLI_PROFILE_NAME>
save_folder=<PATH_TO_ROOT_OF_THIS_REPO>

```


## Step 4:
### Running the Application

Run the Streamlit application using the command provided in each sample's directory for an interactive evaluation experience.

```
streamlit run app.py

```

## How the Evaluation Works

The Model Evaluator leverages an automated approach to assess the performance of AI models, utilizing `anthropic.claude-3-sonnet` as the evaluating model. This section explains the methodology, scale, and criteria used for evaluation.

### Evaluation Methodology

1. **Automated Analysis:** The evaluation process is automated leveraging the `anthropic.claude-3-sonnet` model with Amazon Bedrock. This model analyzes performance data of other models based on predefined criteria.

2. **Data Preparation:** The evaluator processes CSV data containing performance metrics such as 'Total Cost', 'Time Length', and 'Score' of each model being evaluated.

3. **Criteria-Based Scoring:** Each model is scored based on specific evaluation criteria, including faithfulness, answer relevancy, context recall, context precision, context entity recall, answer similarity, answer correctness, harmfulness, maliciousness, coherence, correctness, and conciseness.

4. **Result Summarization:** The evaluation results are summarized to provide a comprehensive overview of each model's strengths and weaknesses across different metrics.

### Evaluation Scale

The scoring for each criterion is done on a scale of 0 to 1 (higher is better)

### Summary Criterion Scoring

1. **Cost Efficiency:** Assesses how well a model manages operational costs relative to its output quality.
   
2. **Speed:** Evaluates the time taken by a model to perform tasks compared to others.
   
3. **Faithfulness:** This measures the factual consistency of the generated answer against the given context. It is calculated from answer and retrieved context. The answer is scaled to (0,1) range. Higher the better.

4. **Answer Relevancy:** Assessing how pertinent the generated answer is to the given prompt. Higher the better. 

5. **Context Recall:** The extent to which the retrieved context aligns with the annotated answer, treated as the ground truth. Higher the better.

6. **Context Precision:** Evaluates whether all of the ground-truth relevant items present in the contexts are ranked higher or not. Higher the better.

7. **Context Entity Recall:** Measure of recall of the retrieved context, based on the number of entities present in both ground_truths and contexts relative to the number of entities present in the ground_truths alone. Higher the better.

8. **Answer Similarity:** Assessment of the semantic resemblance between the generated answer and the ground truth. Higher the better.

9. **Answer Correctness:** Gauging the accuracy of the generated answer when compared to the ground truth. Higher the better. 

10. **Harmfulness:** Binary output to determine if the generated answer contains anything harmfull. 0 means not harmful. 

11. **Maliciousness:** Binary output to determine if the generated answer contains anything malicious. 0 means not malicious. 

12. **Coherence:** Binary output to determine if the generated answer is coherent. 1 means coherent. 

13. **Correctness:** Binary output to determine if the generated answer is correct. 1 means correct. 

14. **Conciseness:** Binary output to determine if the generated answer is concise. 1 means it is concise. 

## Prerequisites
For detailed prerequisites, refer [here](https://github.com/aws-samples/genai-quickstart-pocs#prerequisites).

## Security
For security concerns or contributions, see [CONTRIBUTING](https://github.com/aws-samples/genai-quickstart-pocs/blob/main/CONTRIBUTING.md#security-issue-notifications).

## License
This project is licensed under the MIT-0 License. For more details, see [LICENSE](https://github.com/aws-samples/genai-quickstart-pocs/blob/main/LICENSE).

