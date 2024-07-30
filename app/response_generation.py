from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
import json
import logging

class RephrasedQuestion(BaseModel):
    RephrasedQuestion: str = Field(description="Reframed Question")
    
parser = JsonOutputParser(pydantic_object=RephrasedQuestion)

def generate_response(user_input, context, phi_model, feedback):
    
    prompt_template = PromptTemplate(
    input_variables=["original_question", "context", "feedback"],
    template="""
    You are an AI assistant tasked with reframing questions based on personalized user information. Your goal is to tailor the original question to better suit the user's preferences and background while maintaining its core intent and tone.

    Original question: {original_question}

    User's personalized information:
    #### Personalized Relevant User's Information \n
    {context}

    #### Please follow these steps to reframe the original question:

    1. Analyze the given personalized information and mentally generalize it to broader categories or preferences. 
    For example:
    - Specific food preferences might indicate broader cuisine interests
    - Particular hobbies could suggest general interest areas
    - Mentioned locations might imply lifestyle preferences

    2. Identify key elements of the original question, including its tone, style, and level of formality.

    3. Determine which aspects of the generalized understanding are most applicable to the question.

    4. Integrate the relevant generalized concepts into the original question, without explicitly mentioning the generalization process.

    5. Ensure the CORE intent of the original question is maintained.

    6. Keep the reframed question open-ended and avoid narrowing down to specific options.

    7. Match the tone, style, and level of formality of the original question in the reframed version.

    Important:
    - Use your generalized understanding of ALL relevant personalized information, not just one aspect.
    - Maintain the breadth and simplicity of the original question.
    - Do not assume or add information not implied by the original question or personalized information.
    - Avoid explicitly stating the generalization process in the reframed question.
    - Frame the question from the user's perspective, using "I" and "my" instead of "you" and "your".
    - Keep the reframed question concise and in the same casual tone as the original, if applicable.

    #### Feedback : 
    {feedback}

    {format_instructions}

    Reframed question:
    """,
     partial_variables={"format_instructions": parser.get_format_instructions()},
)

    # Create an LLMChain
    llm_chain = LLMChain(
        llm=phi_model,
        prompt=prompt_template,
        verbose=False
    )

    # Generate response
    response = llm_chain.predict(original_question = user_input, context = context, feedback = feedback)
    try:
        data = json.loads(response)
        return data.get('RephrasedQuestion', user_input)
    except Exception as e:
        logging.info(e)
    
    return user_input    