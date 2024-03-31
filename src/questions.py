import numpy as np
import pandas as pd
import openai
import os
from dotenv import load_dotenv
from openai.embeddings_utils import distances_from_embeddings

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

df = pd.read_csv('processed/embeddings.csv', index_col=0)
# For each row in the embeddings columns, turn into numpy array
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
# Allows to manipulate by flattening embedding into a 1d array

# take users question, take dataframe, and the max length and size, size = embedding model
def create_context(question,df,max_len=1800,size="ada"):
    """
    Create a context for question by finding the most similar context from the dataframe
    We turn the question into an embedding (vectorize) then cross reference the embedded question to embedded data
    Find the closest ones and return them (This part does not include the response from LLM)
    """

    # create embedding for question
    q_embeddings = openai.Embedding.create(
        input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    
    # Create new column in dataframe that calculates the distance of asked question to pertient row in column using cosine method
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # sort by distance and add the text to the context (embedded values retreived from databse) untill the returned list is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add length of the text to the current length
        cur_len += row['n_toekns'] + 4

        # if the context is too long, break
        if cur_len > max_len:
            break

        # else add it to the text that is being returned
        returns.append(row["text"])

    return "\n\n###\n\n".join(returns)
    
# Now we create the function to speak to the LLM, we provide the data from the db and have it interpret it.
def answer_question(df, 
                    model="gpt-3.5-turbo", 
                    question="What is the meaning of life?",
                    max_len=1800,
                    size="ada", # Ada is the embedding model for gpt
                    debug=False,
                    max_tokens=1500,
                    stop_sequence=None):
    """ 
        LLM answers the question based on the most similar retrievals from  the datafram text (Context)
        """
    # Context for example could be the top ten google results for the users question
    context = create_context(
            question,
            df,
            max_len=1800, size=size,
    )
    if debug:
        print("Context is \n:", context)
        print("\n\n")

    try:
        # Create a response from the gpt model
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{
                "role": "user",
                "context": f"I want you to answer the question based on the context below, If you can. If the answer cannot be answered based on the context, say 'I don't know'. \n\n Context: {context}, \n\n --- \n\n Question: {question}"
            }],
            # Below are different parameters that can be tweaked to get different responses
            # Temp = higher = more varried (0-less varried,1)
            temperature=0.5,
            max_tokens=max_tokens,
            # Top is the cummulitive probability that the context used will answer the question?
            top_p = 0.1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
        )
        return response['choices'][0]['messages']['context']
    except Exception as e:
        print(e)
        return ""
    
