import streamlit as st
from pathlib import Path
from knowledge_base_fetcher import fetch_knowledge_bases, get_knowledge_base
from orchestrator import final_evaluator
import os
import boto3
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# set defaults if they are not already set in environment variables
current_directory = os.getcwd()
default_region_name = boto3.DEFAULT_SESSION.region_name if boto3.DEFAULT_SESSION else 'us-east-1'
boto3.setup_default_session()
current_profile_name = boto3.DEFAULT_SESSION.profile_name

os.environ.setdefault('save_folder', current_directory)
os.environ.setdefault('profile_name', current_profile_name)  # Use retrieved or fallback profile name
os.environ.setdefault('region_name', 'default_region_name')

# variables will use their respective values only if they weren't already set elsewhere
save_folder = os.getenv("save_folder")
profile_name = os.getenv("profile_name")
region_name = os.getenv("region_name")

all_knowledge_bases = []
for kb in fetch_knowledge_bases():
    kb_details = get_knowledge_base(kb['knowledgeBaseId'])
    all_knowledge_bases.append(kb_details)

formatted_knowledge_bases = [
    {
        'name': kb['name'],
        'id': kb['knowledgeBaseId'],
        'embedding_model_arn': kb['knowledgeBaseConfiguration']['vectorKnowledgeBaseConfiguration']['embeddingModelArn']
    }
    for kb in all_knowledge_bases
]

# title of the streamlit app
st.title(f""":rainbow[GenAI Embedding Model Evaluator]""")

# default container that houses the document upload field, and model selection box
with st.container():
    # Directions on how to use the application that is shown on the web UI
    st.markdown('Upload a CSV of questions and a CSV of answers, select the knowledge bases you want to compare, and click Evaluate Models')
    # the file upload field, the specific ui element that allows you to upload the file
    questionFile = st.file_uploader('Upload a file of questions', type=["csv"], key="new1")
    # the file upload field, the specific ui element that allows you to upload the file
    answerFile = st.file_uploader('Upload a file of corresponding answers', type=["csv"], key="new2")
    # the model selection box, the specific ui element that allows you to select multiple models
    knowledge_base_options = st.multiselect(
        'Select the knowledge bases you want to compare',
        formatted_knowledge_bases)
    # A button used to trigger the start of a model evaluation job - One button for each job type
    short_form_summary = st.button("Evaluate Models")
    # if the button is clicked for Short form summarization...start the processing
    if short_form_summary:

        # if the file upload field is empty, prevent the user from starting a job
        if questionFile is None or answerFile is None:
            # send a warning message to the user that they need to upload a file
            st.warning("Please upload CSV's to evaluate")
        # if the model selection box is empty, prevent the user from starting a job
        elif len(knowledge_base_options) < 1:
            # send a warning message to the user that they need to select at least one knowledge bases to compare
            st.warning("Please select at least one knowledge bases to compare")
        else:
            # determine the path to temporarily save the CSV file that was uploaded
            save_folder = os.getenv("save_folder")
            # create a posix path of save_folder and the file name
            save_path_1 = Path(save_folder, questionFile.name)
            save_path_2 = Path(save_folder, answerFile.name)
            # write the uploaded CSV to the save_folder you specified
            with open(save_path_1, mode='wb') as w:
                w.write(questionFile.getvalue())
            with open(save_path_2, mode='wb') as w:
                w.write(answerFile.getvalue())
            # once the save path exists...
            if save_path_1 and save_path_2:
                # write a success message saying the file has been successfully saved
                st.success(f'Files {questionFile.name} and {answerFile.name} are successfully saved and Processing.')
                # running the accuracy task, and outputting the results to the front end, that contains a
                # dataframe, the evaluation results, and the cost evaluation results
                st.success(f'Running Evaluations for  {len(knowledge_base_options)} knowledge bases.')
                results_df, evaluation_results, costs_eval_results, score_rubric_df = final_evaluator(save_path_1, save_path_2, 
                                                                                                      knowledge_base_options)  
                st.success(f'Evaluations Completed')
                # Gather cost of the lowest cost model
                lowest_cost = round(results_df['Total Embedding Cost'].min(),8)
                # Gather name of lowest cost model
                lowest_cost_models = results_df[results_df['Total Embedding Cost'] == lowest_cost]['Model'].tolist()
                # Take the list of models name gathered above and cast it to a comma-separated string
                lowest_cost_models_str = ', '.join(lowest_cost_models)
                # Gather the cost of the highest cost model
                highest_cost = round(results_df['Total Embedding Cost'].max(),8)
                # if it is not the lowest cost option, it returns itself
                adjust_highest_cost = lambda x: 0 if x == lowest_cost else x
                # determines the differences between the lowest cost model and highest cost model
                adjusted_highest_cost = round(adjust_highest_cost(highest_cost),2)
                # Find the model with the lowest latency
                shortest_time_length = round(results_df['Time Length'].min(),2)
                # Find the model name of the model with the lowest latency
                shortest_time_length_models = results_df[results_df['Time Length'] == shortest_time_length][
                    'Model'].tolist()
                # Cast the model name as a comma-seperated string
                shortest_time_length_models_str = ', '.join(shortest_time_length_models)
                # find the model with the highest latency
                highest_time = round(results_df['Time Length'].max(),2)
                # if it is not the lowest latency model, return itself
                adjust_highest_time = lambda x: 0 if x == shortest_time_length else x
                # identify the latency of the highest latency model
                adjusted_highest_time = round(adjust_highest_time(highest_time),2)
                # Calculate the highest score from the 'Score' column in the DataFrame df
                highest_score = round(results_df['Score'].max(), 2)
                # Select rows from the DataFrame where the 'Score' is equal to the lowest cost
                # and extract the corresponding 'Model' values into a list
                highest_score_models = results_df[results_df['Score'] == highest_score]['Model'].tolist()
                # Convert the list of lowest cost models to a comma-separated string
                highest_score_models_str = ', '.join(lowest_cost_models)
                # Calculate the lowest score from the 'Score' column in the DataFrame df
                lowest_score = round(results_df['Score'].min(), 2)
                # Define a lambda function to adjust the lowest score
                # It returns 0 if the input x is equal to the highest_score, otherwise it returns x
                adjust_lowest_score = lambda x: 0 if x == highest_score else x
                # Apply the adjust_highest_time function to the lowest_score and convert the result to an integer
                adjusted_lowest_score_str = adjust_lowest_score(lowest_score)
                # Header for model performance graphs
                st.subheader("Model Performance")
                # configure the streamlit front end to have 3 columns
                col1, col2, col3 = st.columns(3)
                # Display a metric in column 1 with a label indicating the lowest cost model(s) and their
                # corresponding cost(s) in ($/1000), Display the current lowest cost value, and the difference
                # between the adjusted highest cost and the lowest cost
                col1.metric(label=f"Lowest Cost Model:",
                            value="{:.6f}".format(lowest_cost), delta=round(adjusted_highest_cost - lowest_cost,2))
                # Display a metric in column 2 with a label indicating the fastest model(s) and their corresponding
                # time(s) in seconds, Display the current shortest time length value, and the difference between the
                # adjusted highest time and the shortest time length
                col2.metric(label=f"Fastest (secs)\n: {shortest_time_length_models_str}", value=shortest_time_length,
                            delta=round(adjusted_highest_time - shortest_time_length,2))
                # Display a metric in column 3 with a label indicating the best result model(s) and their
                # corresponding score(s) in the range 0-1, Display the current highest score value,
                # and the difference between the highest score and the adjusted lowest score
                col3.metric(label=f"Best Results (0-1): \n {highest_score_models_str}", value=highest_score,
                            delta=round(highest_score - adjusted_lowest_score_str, 2))
                # display image containing a graph of the model evaluation results
                st.image('reports/graph.png')
                
                # display a table containing the model ID and the respective evaluation results
                st.write(results_df[['Model', 'Total Embedding Cost(1000)', 'Total LLM Cost(1000)', 'Time Length', 'Score',
                                     'Embedding Character Count', 'LLM Character Count']])
                
                st.subheader("Model Invoke Responses")
                # display a table containing the model ID and the respective output it generated
                st.write(results_df[['Model', 'Invoke Response']].style.set_properties(**{'white-space': 'pre-wrap'}))
                
                
                st.subheader("Model Performance Scores")
                # display scoring rubric
                st.image('reports/rubric_graph.png')
                
                st.write(score_rubric_df)
                # display in markdown a written summary of the cost evaluation results
                st.markdown(costs_eval_results)
                # removing the PDF that was temporarily saved to perform the summarization task
                os.remove(save_path_1)
                os.remove(save_path_2)
