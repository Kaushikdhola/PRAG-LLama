
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
import json
import logging


class Evaluation(BaseModel):
    score: float = Field(description="Score between 0 and 1, where 0 is poor and 1 is excellent")
    reasoning: str = Field(description="A detailed explanation of your score, addressing each of the above criteria and how well the rephrased question performs in each area. Mention any missed opportunities or irrelevant inclusions")
    
parser = JsonOutputParser(pydantic_object=Evaluation)


evaluation_prompt = PromptTemplate(
    input_variables=["Q", "K", "V"],
    template="""
You are an AI evaluator tasked with assessing how well a rephrased question incorporates personalized information and enriches the original question. Your goal is to provide a score and reasoning for the evaluation.

Original question: {Q}

Relevant personalized information:
{K}

Rephrased question: {V}

Please evaluate the rephrased question based on the following criteria:
1. Incorporation of personalized information: How well does the rephrased question integrate the relevant personalized details?
2. Maintenance of original intent: Does the rephrased question preserve the core purpose of the original question?
3. Enrichment: How much does the rephrased question add value or context to the original question?
4. Relevance: Are the incorporated personalized details appropriate and beneficial to the question?
5. Clarity and conciseness: Is the rephrased question clear and concise while including the necessary information?

{format_instructions}

Remember:
- A score of 0 indicates no improvement or relevant incorporation of personalized information.
- A score of 1 indicates perfect incorporation of all relevant personalized information while maintaining the original intent and clarity.
- Be objective and thorough in your evaluation.
- Consider both what was included and what might have been missed from the personalized information.

Evaluation:
""", partial_variables={"format_instructions": parser.get_format_instructions()}
)

def evaluate_response(Q, K, V, model):
    
    judge_model = LLMChain(
        llm=model,
        prompt=evaluation_prompt,
        verbose=False
    )
    
    eval_result = judge_model.predict( Q = Q, K = K, V = V)
    

    try:
        data = json.loads(eval_result)
        score = data.get('score', 0)
        reasoning = data.get('reasoning', "")
        print(score)
        print(reasoning)
        return score, reasoning
    except Exception as e:
        logging.info(e)
    
    return 0, ""

    
