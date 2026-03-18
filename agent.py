# import libraries

import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub

# import tools
from my_tools import find_stock_ticker, get_stock_data, arxiv_search, stock_news_search, stock_website_search

load_dotenv()

gemini_api_key = os.getenv('gemini_api_key')


# prompt
system_prompt = """
You are a professional AI Stock Market Assistant.

Available tools:
- find_stock_ticker → find the ticker symbol for a company name
- real_time_stock_data → get stock price and financial data using a ticker symbol
- get_stock_news → get recent news about a company or stock
- arxiv_research_papers → find academic research papers about finance or economics
- stock_website_search → search financial concepts and explanations

Rules:
1. If the user provides a company name instead of a ticker, first use the find_stock_ticker tool.
2. Once you have the ticker symbol, use the real_time_stock_data tool to get stock information.
3. If the question asks for recent news, use the get_stock_news tool.
4. If the question asks for academic research or papers, use the arxiv_research_papers tool.
5. If the question asks for general financial knowledge or explanations, use the stock_website_search tool.
6. Always use the most relevant tool when information is required.
7. Provide clear, concise answers based on the tool results.

Important:
- real_time_stock_data requires a ticker symbol, not a company name.
- If the ticker is unknown, use find_stock_ticker first.
"""

# llm
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",
                             google_api_key=gemini_api_key,
                             temperature=0.7,
                             max_output_tokens=1024,
                             max_retries=2,
                             timeout=60)


# prompt template
prompt = hub.pull("hwchase17/react-chat")

# final prompt
prompt.template = f"""
                    {system_prompt}

                    """ + prompt.template


# tools
tools = [
    find_stock_ticker, get_stock_data, arxiv_search, stock_news_search,
    stock_website_search
]


# create agent and agent executor
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent,
                               tools=tools,
                               verbose=True,
                               handle_parsing_errors=True,
                               max_iterations=5)


# function which generate final responses by using chat history and tools
def run_stock_agent(query: str, chat_history=None):

    if chat_history is None:
        chat_history = []

    try:

        response = agent_executor.invoke({
            "input": query,
            "chat_history": chat_history
        })

        return response["output"]

    except Exception as e:

        if "RESOURCE_EXHAUSTED" in str(e):
            return "API quota exceeded. Please try again later."

        print(e)

        return "Something went wrong while analyzing the request."
