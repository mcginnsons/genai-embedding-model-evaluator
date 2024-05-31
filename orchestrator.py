import os
import boto3
from AnthropicTokenCounter import AnthropicTokenCounter
from text_extractor_and_summarizer import (text_extraction, text_formatter)
from orchestration_helper import OrchestrationHelper
from pricing_calculator import calculate_total_price
from evaluation_steps import evaluate_model_output, evaluate_model_performance
from plotting_and_reporting import write_evaluation_results, plot_model_comparisons, plot_model_performance_comparisons
import logging
from timeit import default_timer as timer
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws.retrievers.bedrock import AmazonKnowledgeBasesRetriever
from langchain.chains import RetrievalQA
from datasets import Dataset
## TODO add langchain, langchain_aws, datasets and ragas to requirements.txt
from ragas import evaluate
from ragas.metrics import (
    faithfulness, 
    answer_relevancy, 
    context_recall, 
    context_precision, 
    context_entity_recall, 
    answer_similarity, 
    answer_correctness
    )
from ragas.metrics.critique import (
    harmfulness, 
    maliciousness, 
    coherence, 
    correctness, 
    conciseness
    )
# Loading environment variables from a .env file.
load_dotenv()
# logging setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Configuring the default AWS session using the provided profile name from environment variables.
boto3.setup_default_session(profile_name=os.getenv("profile_name"))
# Bedrock Runtime client used to invoke and question the models
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv("region_name")
)
llm_for_text_generation = ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0", client=bedrock_runtime) ##TODO update to allow user to select the model they want to use
llm_for_evaluation = ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0", client=bedrock_runtime)

metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_entity_recall,
        answer_similarity,
        answer_correctness,
        harmfulness, 
        maliciousness, 
        coherence, 
        correctness, 
        conciseness
    ]


def final_evaluator(csv_path_1, csv_path_2, knowledge_bases):
    """
    Evaluate multiple knowledge bases for accuracy and other evaluation metrics.

    :param csv_path_1: Path to the question CSV file.
    :param csv_path_2: Path to the answer CSV file.
    :param knowledge_bases: List of knowledge bases to evaluate.

    :return: A tuple containing:
        - DataFrame: Evaluation results including knowledge base performance metrics and costs.
        - str: Summary of the evaluation results.
        - str: Evaluation of the costs for model selection.
        - DataFrame: Scoring rubric for the evaluated models.
    """
    # Extract the questions out of the given CSV
    questions = text_extraction(csv_path_1)
    # Extract the answers out of the given CSV
    ground_truths = text_extraction(csv_path_2)

    # Calculate the character count of the input text
    embedding_character_count = len(text_formatter(questions))

    # Initialize an empty list to store results for each knowledge base
    results_list = []
    # Initialize an empty list to store the scoring rubric for each model
    score_rubric_list = []
    # Initialize an empty string to store the aggregated evaluation results
    evaluation_results = ""
    # for each mode evaluate 
    for knowledge_base in knowledge_bases:

        embedding_model_name = knowledge_base['embedding_model_arn'].split('/')[1]

        bedrock_embeddings = BedrockEmbeddings(model_id=embedding_model_name, client=bedrock_runtime)
        retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=knowledge_base['id'],
            retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}}
        )

        input_embedding_token_count = len(text_formatter(questions))/6 ## TODO FIX ME!!!

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_for_text_generation, retriever=retriever, return_source_documents=True
        )

        answers = []
        contexts = []
        token_counter = AnthropicTokenCounter(llm_for_text_generation)
        input_llm_token_count = 0
        output_llm_token_count = 0

        start = timer()
        for question in questions:
            answers.append(qa_chain.invoke(question, config={"callbacks": [token_counter]})["result"])
            input_llm_token_count += token_counter.input_tokens
            output_llm_token_count += token_counter.output_tokens
            contexts.append([docs.page_content for docs in retriever.get_relevant_documents(question)])
        # end timer
        end = timer()
        # calculate total time taken
        time_length = round(end - start, 2)
        # calculate time taken per character
        char_process_time = embedding_character_count / time_length
        # calculate llm_character_count
        llm_character_count = embedding_character_count + len(" ".join(item[0] for item in contexts))

        # To dict
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        # Convert dict to dataset
        dataset = Dataset.from_dict(data)
        # Run RAGAS on dataset
        result = evaluate(
            dataset = dataset, 
            metrics=metrics,
            llm=llm_for_evaluation,
            embeddings=bedrock_embeddings,
        )

        input_embedding_cost, output_embedding_cost, embedding_total_cost, embedding_total_cost_1000 = calculate_total_price(input_embedding_token_count, 0, embedding_model_name)
        input_llm_cost, output_llm_cost, llm_total_cost, llm_total_cost_1000 = calculate_total_price(input_llm_token_count, output_llm_token_count, llm_for_text_generation.model_id)

        # evaluate the models performance against the grading rubric and return the final score, final summary
        # and the scoring rubric
        final_score, final_summary, final_score_rubric = evaluate_model_output(result, knowledge_base, contexts)
    
        #  create a OrchestrationHelper object to store the results of the evaluation
        result = OrchestrationHelper(knowledge_base['name'], time_length, embedding_character_count, llm_character_count, char_process_time, input_embedding_cost, 
                                     output_embedding_cost, embedding_total_cost, embedding_total_cost_1000, input_llm_cost, 
                                     output_llm_cost, llm_total_cost, llm_total_cost_1000,
                                     final_score, text_formatter(answers), final_summary)
        # add the results of the evaluation to the results list
        results_list.append(result.format())
        # add the evaluation results written summary to the evaluation results string
        evaluation_results += result.evaluation_results()            
        # add the scoring rubric for the model into the scoring rubric list
        score_rubric_list.append(final_score_rubric)

    # Setting the display to max column width
    pd.set_option('display.max_colwidth', None)
    # Convert scoring rubric list into a DataFrame
    score_rubric_df = pd.DataFrame(score_rubric_list)
    # Convert performance and cost results list into a DataFrame
    results_df = pd.DataFrame(results_list)
    # Save this dataframe as a CSV file      
    file_path = os.path.join('reports', 'model_performance_comparison.csv')
    # Save DataFrame to CSV file
    results_df.to_csv(file_path, index=False)
    # Convert DataFrame to CSV format string to send to Bedrock for eval
    csv_data = StringIO()
    # Save DataFrame to CSV file
    results_df.to_csv(csv_data, index=False)
    # Move to start of StringIO object to read its content
    csv_data.seek(0)
    # Read CSV data from StringIO object (as a string)
    csv_string = csv_data.getvalue()
    # ask the model which is the best model to use for cost and performance
    invoke_costs_eval_response = evaluate_model_performance(csv_string, "anthropic.claude-3-sonnet-20240229-v1:0")
    # chart out the performance and cost results
    plot_model_comparisons(results_df)
    # plot the performance rubric scores 
    plot_model_performance_comparisons(score_rubric_df)
    # Save the reports to a file
    write_evaluation_results(evaluation_results, eval_name="summary")
    write_evaluation_results(invoke_costs_eval_response, eval_name="cost")
    #  return the results dataframe, evaluation results, invoke costs eval response and score rubric dataframe
    return results_df, evaluation_results, invoke_costs_eval_response, score_rubric_df
