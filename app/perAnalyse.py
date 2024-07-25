from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import FAISS
import json

def personalize_analyzer(text, phi_model, vector_store, embeddings):

    prompt_template = """

    ### Personalized Content Analyzer

    You are an AI assistant trained to identify personalized content in text. Personalized content includes any information that is specific 
    to an individual user, such as their preferences, history, demographics, or any identifiable information. Your task is to analyze the 
    given text and determine if it contains any personalized content.

    #### Instructions:
    Analyze the following text to determine if it contains any personalized content. 
    Personalized content includes, but is not limited to:

    1. Individual preferences (e.g., "I like", "I prefer", "My favorite", "I love")
    2. Personal history or experiences (e.g., "I've been to", "I've watched", "Last week I", "I work")
    3. Demographic information (e.g., age, gender, location, occupation)
    4. Specific identifiers (e.g., names, usernames, email addresses)
    5. Individual circumstances (e.g., "My car broke down", "I'm looking for a new job")
    6. Financial information (e.g., "My budget is", "I can afford")
    7. Educational background (e.g., "I studied", "My major was")
    8. Family or relationship details (e.g., "My spouse", "My kids")
    9. Personal goals or intentions (e.g., "I want to", "I'm planning to")
    ---

    Text to analyze: {text}
    
    ---
    #### Step-by-step analysis: 
    1. Identify any words or phrases that indicate personal information (e.g., "I", "my", "me").
    2. Look for specific examples of preferences, experiences, or circumstances.
    3. Check for any demographic or identifiable information.
    4. Determine if the text is written from a personal perspective or contains individual-specific details.
    
    Based on the analysis, provide a response indicating whether the text contains personalized content. If personalized content is found, break down and summarize each piece of 
    personalized information in a single line ONLY no need to give any explaination.

    You MUST give formatted response here in JSON with the following format ONLY:
    ``` 
    JSON

    {{
        "isPersonalized": "yes/no",
        "personalizedContent": ["summarized personalized content 1", "summarized personalized content 2", ...]
    }}
    ```
    Ensure each piece of personalized content is summarized in a single, concise line.
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
        try: 
            return json.loads(generated_text)

        except Exception as e:
            return json.loads('{"isPersonalized": "no", "personalizedContent": []}')
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return json.loads('{"isPersonalized": "no", "personalizedContent": []}')