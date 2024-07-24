from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import FAISS

def personalize_analyzer(text, phi_model, vector_store, embeddings):
    prompt_template = """
    Analyze the following text to determine if it contains any personalized content. 
    Personalized content includes, but is not limited to:

    1. Individual preferences (e.g., "I like", "I prefer", "My favorite")
    2. Personal history or experiences (e.g., "I've been to", "I've watched", "Last week I")
    3. Demographic information (e.g., age, gender, location, occupation)
    4. Specific identifiers (e.g., names, usernames, email addresses)
    5. Individual circumstances (e.g., "My car broke down", "I'm looking for a new job")
    6. Health-related information (e.g., "I'm allergic to", "My doctor said")
    7. Financial information (e.g., "My budget is", "I can afford")
    8. Educational background (e.g., "I studied", "My major was")
    9. Family or relationship details (e.g., "My spouse", "My kids")
    10. Personal goals or intentions (e.g., "I want to", "I'm planning to")

    Text to analyze: {text}

    Step-by-step analysis:
    1. Identify any words or phrases that indicate personal information (e.g., "I", "my", "me").
    2. Look for specific examples of preferences, experiences, or circumstances.
    3. Check for any demographic or identifiable information.
    4. Determine if the text is written from a personal perspective or contains individual-specific details.

    Based on your analysis, respond with ONLY ONE of the following:
    - If ANY personalized content is found, respond with "yes".
    - If NO personalized content is found, respond with "no".

    Your one-word response:
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