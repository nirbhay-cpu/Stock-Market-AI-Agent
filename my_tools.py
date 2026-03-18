# import libraries

import os
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.tools import tool

import yfinance as yf
from apify_client import ApifyClient

from langchain_community.utilities import ArxivAPIWrapper
import feedparser
from urllib.parse import quote
from bs4 import BeautifulSoup

from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# --- Tool 1 : Ticker Finder  ---

@tool("find_stock_ticker")
def find_stock_ticker(company_name: str) -> str:
    """
    Find the stock ticker symbol for a given company name.
    Use this tool when the user provides a company name instead of a ticker.
    """

    try:
        results = yf.Search(company_name)
        quotes = results.quotes

        if not quotes:
            return f"No ticker symbol found for {company_name}"

        for q in quotes:
            if q.get("quoteType") == "EQUITY":
                ticker = q.get("symbol")
                name = q.get("shortname")
                return f"Ticker: {ticker}, Company: {name}"

        ticker = quotes[0]["symbol"]
        name = quotes[0].get("shortname", "Unknown")

        return f"Ticker: {ticker}, Company: {name}"

    except Exception as e:
        return f"Error finding ticker: {str(e)}"
    
    
# --- Tool 2 : Real time Stock data  ---


client = ApifyClient(os.getenv("APIFY_API_TOKEN"))


@tool("real_time_stock_data")
def get_stock_data(symbol: str) -> dict:
    """Retrieve real-time stock market data for publicly traded companies.
    Input should be a stock ticker symbol or company name.
    """
    try:
        run = client.actor("automation-lab/yahoo-finance-scraper").call(
            run_input={"tickers": [symbol]})

        dataset_items = client.dataset(
            run["defaultDatasetId"]).list_items().items

        if not dataset_items:
            return {"error": f"No data found for {symbol}"}

        return dataset_items[0]

    except Exception as e:
        return {"error": str(e)}


# --- Tool 3 : Get Research Paper  ---


arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=2000)


@tool("arxiv_research_papers")
def arxiv_search(query: str) -> str:
    """Search ArXiv for academic research papers related to finance, stock markets,
        trading strategies, quantitative finance, algorithmic trading, and financial
        machine learning. Use this tool when the user asks for research studies or
        academic publications."""

    docs = arxiv_wrapper.get_summaries_as_docs(
        f"{query} stock market finance trading")

    if not docs:
        return "No research papers found."

    doc = docs[0]

    return f"""
                Title: {doc.metadata['Title']}
                Authors: {doc.metadata['Authors']}
                Published: {doc.metadata['Published']}
                Summary: {doc.page_content}
                Paper Link: {doc.metadata['Entry ID']}
            """

# --- Tool 4 : Get Stock News  ---


@tool("get_stock_news")
def stock_news_search(company: str) -> str:
    """
        Search recent news articles about companies, stock markets,
        finance, and economic events. Use this tool when the user asks 
        about recent news, market updates, or company developments.
    """
    query = quote(company + " stock news")

    url = f"https://news.google.com/rss/search?q={query}+stock&hl=en-US&gl=US&ceid=US:en"

    feed = feedparser.parse(url)

    news = []

    for entry in feed.entries[:5]:

        summary = BeautifulSoup(entry.summary, "html.parser").text
        published = datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d")
        source = entry.source["title"] if "source" in entry else "Unknown"

        news.append(
            f"Title: {entry.title}\nSummary: {summary}\nPublished: {published}\nSource: {source}\nLink: {entry.link}\n"
        )

    return "\n".join(news)


# --- Tool 5 : Stock Terminologies  ---


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@tool("stock_website_search")
def stock_website_search(query: str) -> str:
    """
    Search stock related information from a website using a Chroma vector database.
    If the database does not exist, it will create it.
    """

    gemini_api_key = os.getenv('gemini_api_key')
    db_path = "./stock_faiss_db_0"
    url = "https://en.wikipedia.org/wiki/Stock_market"

    # check if vector db exists
    if os.path.exists(db_path):

        vector_db = FAISS.load_local(db_path,
                                     embeddings,
                                     allow_dangerous_deserialization=True)

    else:
        print("Creating vector database...")

        # load website
        loader = RecursiveUrlLoader(url=url,
                                    max_depth=1,
                                    headers={"User-Agent": "Mozilla/5.0"})

        docs = loader.load()

        # split docs
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                                  chunk_overlap=200)

        split_docs = splitter.split_documents(docs)

        # create vector db
        vector_db = FAISS.from_documents(split_docs, embeddings)

        vector_db.save_local(db_path)

    # create retriever
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    print("Extracting data from vectordb")
    results = retriever.invoke(query)

    formatted_results = []

    for doc in results:
        clean_text = BeautifulSoup(doc.page_content, "html.parser").get_text()

        formatted_results.append(f"""
                    Title: {doc.metadata.get('title')}
                    Source: {doc.metadata.get('source')}

                    Content:
                    {clean_text[:500]}
                    """)

    return "\n".join(formatted_results)
