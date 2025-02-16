from flask import Flask, request, render_template
from langchain_core.chat_history import BaseChatMessageHistory
from markdown_it import MarkdownIt
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import Tool
from langchain_core.output_parsers import StrOutputParser
from duckduckgo_search import DDGS
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

search = DuckDuckGoSearchResults(name="search")
tools = [search]
parser = StrOutputParser()
llm = ChatGroq(model="llama-3.3-70b-versatile")

def duckduckgo_search(query):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)  # Get top 5 results
    return "\n".join([f"{r['title']} - {r['href']}\nSnippet: {r['body']}" for r in results])

search_tool = Tool(
    name="DuckDuckGo Search",
    func=duckduckgo_search,
    description="Use this tool to search the internet and fetch real-time information when needed."
)

system = (
    "You are a Startup Assistant Chat bot, Answer the questions asked by the user based on the chat history. "
    "The user is a software developer and trying to create a startup after leaving the job. "
    "Suggest the appropriate actions to be taken by the user. "
    "Search the internet and give the best answer to the user's question, also give links to the resources. "
    "Also the chat history will be given, give appropriate answer based on it."
)

prompt = ChatPromptTemplate(
    [
        ("system", system),
        ("human", "{input},{history}")
    ]
)

def system2():
    system2 = (
        "You are a web researcher, the user is trying to validate their software startup idea. "
        "Based on the user query, make 4 Search Engine Queries such that it tries to find similar websites or software and search according to them. "
        "Then based on the metadata of the websites displayed validate the idea provided by the user."
    )
    prompt2 = ChatPromptTemplate([
        ("system", system2),
        ("human", "{input}")
    ])
    return prompt2

def get_chain():
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

def get_chain2():
    prompt2 = system2()
    chain2 = LLMChain(llm=llm, prompt=prompt2)
    return chain2

app = Flask(__name__)

Chat_history = []

def clear():
    global Chat_history
    Chat_history = []

content = []
md = MarkdownIt()

@app.route("/", methods=["GET", "POST"])
def Chatot():
    global content

    if request.method == "POST":
        input = request.form.get("input")
        new_chat = request.form.get("new_chat")
        validate_idea = request.form.get("validate_idea")

        if new_chat:
            clear()
            content = []
            return render_template("pagebase.html")

        if input:
            if validate_idea == "true":  # Ensure the value is correctly checked
                chain = get_chain2()
            else:
                chain = get_chain()
            response = chain.invoke({"input": input, "history": Chat_history})
            parsed = md.render(response["text"])  # Access the 'text' key in the dictionary
            content = [parsed, input]
            Chat_history.append((input, parsed))
            return render_template("page.html", content=content, history=Chat_history)

    return render_template("pagebase.html")

if __name__ == "__main__":
    app.run(debug=True)