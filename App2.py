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
from langchain.agents import create_openai_tools_agent
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.chat import  MessagesPlaceholder
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
        results = ddgs.text(query, max_results=3)  # Get top 5 results
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
        ("human", "{input},{history}"),
        MessagesPlaceholder("agent_scratchpad")
    ]
)
def system2():
    system2 = (

        """You are a web researcher, the user is trying to validate his software startup idea, Based on the user query, 
        make 3 Search Engine Queries such that it tries to find similar websites or software and search according to them
        Then based on the metadata of the websites displayed validate the idea provided by the user
        also give the links to these websites,your main task is to find out if the users idea is actually valid and can be successful or not
        so focus more on this aspect rather than the metadata of the websites"""
    )
    prompt2 = ChatPromptTemplate(
        [
            ("system",system2),
            ("human","{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ]
    )
    return prompt2

def get_chain():
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    chain = prompt|llm
    # Initialize the agent with tool-calling capabilities
    agentic=create_openai_tools_agent(llm=llm,prompt=prompt,tools=[search_tool])
    agent1 = AgentExecutor(agent=agentic,memory=memory,tools=[search_tool],verbose=True)
    return agent1
def get_chain2():
    prompt2=system2()
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    chain2 = prompt2|llm
    # Initialize the agent with tool-calling capabilities
    agentic2 = create_openai_tools_agent(llm=llm,tools=[search_tool],prompt=prompt2)
    agent2 = AgentExecutor(agent=agentic2,memory=memory,tools=[search_tool],verbose=True)
    return agent2

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
            # parsed = md.render(response["content"])  # Access the 'text' key in the dictionary
            content = [response, input]
            Chat_history.append((input, response))
            return render_template("page.html", content=content, history=Chat_history)

    return render_template("pagebase.html")

if __name__ == "__main__":
    app.run(debug=True)