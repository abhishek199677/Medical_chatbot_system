from euriai.langchain import create_chat_model

# sets up AI chat interface
def get_chat_model(api_key):   #function creates and returns a chat model object and  can use to talk to a large language model (LLM)
    chat_model = create_chat_model(
        api_key=api_key,       
        model="gpt-4.1-nano",  
        temperature=0.7     #Controls randomness of the responses
    )
    return chat_model  
    

# Sends questions and gets answers from the AI.
def ask_chat_model(chat_model, prompt: str):
    response = chat_model.invoke(prompt) #invoke is a function that actually talks to the LLM
    return response.content







