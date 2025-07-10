
def initialize_llm(model_name="gpt-4o-mini", temperature=0.0, max_tokens=4096):
    """
    Initialize the LLM model with specified parameters.
    Args:
        model_name (str): The name of the LLM model to use.
        temperature (float): Sampling temperature for the model.
        max_tokens (int): Maximum number of tokens to generate.
    Returns:
        ChatOpenAI: An instance of the ChatOpenAI model.
    """
    if 'gpt' in model_name:
        # Ensure the model name is compatible with OpenAI's API
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    elif 'gemini' in model_name:
        # Initialize Gemini model (if applicable)
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Supported models are 'gpt' and 'gemini'.")

def parse_llm_response(response):
    """
    Parse the LLM response object to extract the main content as a string.
    Handles OpenAI, LangChain, and dict response formats.
    """
    if hasattr(response, 'content'):
        return response.content.strip()
    elif isinstance(response, dict):
        # OpenAI format
        if 'choices' in response and response['choices']:
            if 'text' in response['choices'][0]:
                return response['choices'][0]['text'].strip()
            elif 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
                return response['choices'][0]['message']['content'].strip()
        elif 'content' in response:
            return response['content'].strip()
    return str(response).strip()

def handle_agent_response(agent_name, response):
    """
    Handle and structure the LLM response according to the agent's expected output.
    Args:
        agent_name (str): The name of the agent (e.g., 'supervisor', 'summarizer', etc.)
        response: The raw LLM response object.
    Returns:
        dict: Structured output for the agent node.
    """
    content = parse_llm_response(response)
    if agent_name == "supervisor":
        return {"update": {"supervisor_directives": [content], "progress_log": ["Supervisor issued directive."]}}
    elif agent_name == "summarizer":
        return {"update": {"artifacts": {"summary": content}, "progress_log": ["Summarizer completed."]}}
    elif agent_name == "gap_finder":
        return {"update": {"artifacts": {"gaps": content}, "progress_log": ["Gap analysis completed."]}}
    elif agent_name == "synthesizer_writer":
        return {"update": {"artifacts": {"synthesis": content}, "progress_log": ["Synthesizer/Writer completed synthesis."]}}
    elif agent_name == "literature_search":
        # This agent may need both results and summary, so content is summary, results should be passed separately
        return {"update": {"artifacts": {"literature_summary": content}, "progress_log": ["Literature search and summary completed."]}}
    else:
        # Default: just return the parsed content
        return {"update": {"content": content}}

def invoke_llm(llm, prompt):
    """
    Invoke the LLM with a given prompt and return the parsed response.
    Args:
        llm (ChatOpenAI): The LLM model instance.
        prompt (str): The prompt to send to the model.
    Returns:
        str: The parsed response from the LLM.
    """
    response = llm.invoke(prompt)
    return parse_llm_response(response)
