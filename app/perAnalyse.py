from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import FAISS
import json

def personalize_analyzer(text, phi_model):

    prompt_template = """### Personalized Content Analyzer

You are an AI assistant trained to identify personalized content in text. Your task is to analyze the given text and determine if it contains any personalized content about the user.

#### Instructions:
Analyze the following text to determine if it contains any personalized content about the user. 
Extract only factual, explicitly stated information about:

1. Personal characteristics or identifiers
2. Preferences or interests
3. Habits or routines
4. Skills or abilities
5. Experiences or history
6. Current circumstances
7. Future plans or goals
8. Relationships or social connections
9. Opinions or beliefs
10. Any other explicitly stated user-specific details

**IMPORTANT**: 
- List each piece of personalized information as a separate, concise element.
- Do not add interpretations, explanations, or inferences.
- Include only information explicitly stated in the text.


#### Examples:

Example 1:
Text: "The sky is blue."
Output in JSON:
"isPersonalized": "no",
"personalizedContent": []


Example 2:
Text: "I enjoy reading science fiction novels before bed."
Output in JSON:
"isPersonalized": "yes",
"personalizedContent": [
"Enjoys reading science fiction",
"Has a habit of reading before bed"
]


Example 3:
Text: "As someone who's allergic to peanuts, I always check food labels carefully."
Output in JSON:
"isPersonalized": "yes",
"personalizedContent": [
"Has a peanut allergy",
"Habitually checks food labels"
]
---

Text to analyze: {text}

---
#### Analysis steps: 
1. Scan for any first-person pronouns or statements that indicate personal information.
2. Identify any specific details about the user's life, preferences, or circumstances.
3. Look for patterns or repeated behaviors that might indicate habits or routines.
4. Consider any information that distinguishes this user from others.

Provide a response indicating whether the text contains personalized content about the user. If personalized content is found, summarize each piece of information concisely.

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
        verbose=False,
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