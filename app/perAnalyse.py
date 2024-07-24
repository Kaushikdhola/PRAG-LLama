from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import FAISS

def personalize_analyzer(text, phi_model, vector_store, embeddings):
    prompt_template = """
    Analyze the following text and determine if it contains any personalized content.
    Personalized content refers to any information that is specific to an individual user,
    such as their preferences, history, demographics, or any identifiable information.

    Text: {text}

    If the text contains personalized content, respond with only the word "yes".
    If no personalized content is found, respond with only the word "no".

    Response:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm_chain = LLMChain(
        llm=phi_model,
        prompt=prompt,
        verbose=True,
        callback_manager=callback_manager
    )

    try:
        response = llm_chain.generate([{"text": text}])
        
        generated_text = response.generations[0][0].text.strip().lower()
        
        print("Personalized content analysis response:", generated_text)
        
        if "yes" in generated_text:
            return "yes"
        else:
            return "no"
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return "no"