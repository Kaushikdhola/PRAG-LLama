from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import logging
import json
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

class Question(BaseModel):
    Questions: list = Field(description="list of questions")
    
parser = JsonOutputParser(pydantic_object=Question)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    
    template="""
    ### Multi Query Assistant
    You are an AI language model assistant. Your task is to generate FIVE 
    different versions of the given user question to retrieve relevant documents from a vector database.
    Use WH (Why, What, Which, How) questions to find personalized context relevant to the original question. 
    Incorporate user-specific information when available to make the questions more tailored and relevant.
    By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations 
    of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    
    #### Step-by-step thought process: 
    1. Analyze the original question to understand the main topic and intent.
    2. Identify potential areas where personal context could enhance the answer.
    3. Formulate questions that would elicit relevant personal information.
    4. Ensure questions are diverse and cover different aspects of personalization.
    5. Limit to 5 questions, focusing on those most likely to improve the answer quality.
    
    #### Example:
        1. Original question: Code of fibonacci series
            Q1. What programming language does the user prefer for implementing the Fibonacci series?
            Q2. Which approach does the user prefer for the Fibonacci series: recursive or iterative?
            Q3. Does the user prefer concise or detailed explanations for the Fibonacci series code?
            Q4. Should examples of sample output be included in the Fibonacci series explanation?
            Q5. Does the user require pseudocode for the Fibonacci series implementation?
            
        2. Original question: Best places to visit in Europe
            Q1. What type of traveler are you (e.g., history buff, foodie, nature lover, party-goer)?
            Q2. What is your budget range for this European trip?
            Q3. How long do you plan to stay in Europe?
            Q4. Do you have any dietary restrictions or preferences that might influence your destination choices?
            Q5. Are you interested in popular tourist spots or off-the-beaten-path destinations?
    
    
    Give upto 5 questions only (WITHOUT ANY EXPLAINATION) if the question is able to capture personalized information using the semantic search from
    the vector database to answer the original question in an effective way.    
    
    You MUST give formatted response here in JSON.
    
    {format_instructions}
    ---
    Original question: {question}
    """,
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

def multiQueryRetreiver(user_query, phi_model):
    
    llm_chain = LLMChain(
        llm=phi_model,
        prompt=QUERY_PROMPT,
        verbose=False
    )
    
    response = llm_chain.predict(question = user_query)
    
    queries = [user_query]

    try:
        data = json.loads(response)

        questions = data.get('Questions', [])
        queries.extend(questions)
    except Exception as e:
        logging.info(e)
    
    return queries