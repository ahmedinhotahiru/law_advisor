#-------------------------------- IMPORT LIBRARIES --------------------------------

import chromadb # For chroma database
from langchain_community.vectorstores import Chroma # For chroma vectorstore
from langchain_openai import OpenAIEmbeddings # To use OpenAI for embedding the RAG vectorstore contents
from langchain_openai import ChatOpenAI # To use ChatGPT as the Base LLM

# Need to pip install pypdf for this to work
from langchain_community.document_loaders import PyPDFLoader # For loading and parsing PDF documents
from langchain_text_splitters import RecursiveCharacterTextSplitter # For splitting extracted text into chunks

from langchain.tools.retriever import create_retriever_tool # For creating vectorstore retriever tools
from langchain_core.prompts import ChatPromptTemplate # For defining chatagent prompt
from langchain_core.prompts import MessagesPlaceholder # Prompt template for defining chatagent prompt

from langchain_community.tools.tavily_search import TavilySearchResults

from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages # To convert (AgentAction, tool output) tuples into ToolMessages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser # To Parse a message into agent actions/finish
from langchain.agents import AgentExecutor # To use an agent that uses tools

from langchain_core.messages import AIMessage, HumanMessage # For AI and HumanMessage interactions

import chainlit as cl # To use chainlit

import os # For operating system related actions
from tqdm import tqdm # For displaying progress bars
from dotenv import load_dotenv



#---------------- Load environment variables ----------------
load_dotenv()




#---------------- Define objects needed to initialize vectorstore ----------------
embeddings = OpenAIEmbeddings(openai_api_key = "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")
text_splitter = RecursiveCharacterTextSplitter() # For splitting extracted text into chunks


#---------------- FUNCTION TO INITIALIZE OR LOAD VECTORSTORE ----------------
def init_vectorstore(collection_name="business_law_docs", pdf_dir="docs", db_path="./chroma_db"):

    """
    Args:
        collection_name (str): The name of the chroma collection/table
        pdf_dir (str); Directory pf pdf documents

    Returns:
        chroma: Vectorstore
    """

    # Initialize the chromadb persistent client from the db path
    persistent_client = chromadb.PersistentClient(path=db_path)

    # Now initialize or load this chromadb collection/table
    # If the collection/table exists already, load it, otherwise create it
    collection = persistent_client.get_or_create_collection(collection_name)

    # Initialize or load the vectorstore with specified collection name
    # Also specify the persistent client to be able to find the collection
    # Lastly, soecify wha embedding function to use for embedding the vectorstore
    vector_store = Chroma(
        collection_name=collection_name, 
        client=persistent_client, 
        embedding_function=embeddings
    )
    

    # If the collection is empty, extract text from documents and load into the collection
    if collection.count() == 0:

        # Display loading message to user
        print(f"\nLoading documents into collection {collection_name}")

        # Define list of pdf file(name)s to extract text from
        pdf_files = [file for file in os.listdir(pdf_dir) if file.endswith('.pdf')]

        # Check the number of documents we want to extract from
        num_docs = len(pdf_files)

        # Container to hold extracted tect from documents
        all_extracted_text = []

        # Loop through all documents while extracting text from them
        # use tqdm to display progress bar of loading files

        with tqdm(total=num_docs, desc="Extracting text from PDF documents") as progress_bar:

            # Loop through each file
            for file_name in pdf_files:

                # Define filepath
                file_path = os.path.join(pdf_dir, file_name)

                # Load the PDF using the PDF loader library
                loaded_pdf = PyPDFLoader(file_path=file_path)

                # Extract text from the loaded PDF file
                extracted_text = loaded_pdf.load()

                # Append the extracted text to the general container outside by extending
                all_extracted_text.extend(extracted_text)

                # Update the progress bar
                progress_bar.update(1)


        # Now split all the the extracted documents into chunks with some level of overlap
        print("\nSplitting extracted text into chunks...")
        splitted_text = text_splitter.split_documents(all_extracted_text)

        # Add splitted text chunks to the initialized vectorstore
        # They automatically get embnedding by the vectorstore's embedding function, 
        # which is openai embeddings in this case
        vector_store.add_documents(splitted_text)

        # Display success message
        print(f"\nNew collection ({collection_name} created and populated successfully)")


    else:
        # The collection already exists so load it
        print(f"\nLoading existing collection ({collection_name})")

    return vector_store




#---------------- Define variables to initialize vectorstore ----------------

# Directory of pdf files
general_dir = "./docs/General Laws"
bank_dir = "./docs/Bank Laws"
company_dir = "./docs/Company Laws"
electronic_dir = "./docs/Electronic Laws"
immigration_dir = "./docs/Immigration and Emigration"
land_property_dir = "./docs/Land and Property"
tax_dir = "./docs/Land and Property"

# Define chromadb collection name
general_laws_collection = "general_laws"
bank_laws_collection = "bank_laws"
company_laws_collection = "company_laws"
electronic_laws_collection = "electronic_laws"
immigration_laws_collection = "immigration_laws"
land_property_laws_collection = "land_property_laws"
tax_laws_collection = "land_property_laws"

# Path to chroma database
db_path = "./chroma_db"



#---------------- Create new or load an existing vectorstore(s) ----------------
law_vector = init_vectorstore(general_laws_collection, general_dir, db_path)
bank_law_vector = init_vectorstore(bank_laws_collection, bank_dir, db_path)
company_law_vector = init_vectorstore(company_laws_collection, company_dir, db_path)
electronic_law_vector = init_vectorstore(electronic_laws_collection, electronic_dir, db_path)
immigration_law_vector = init_vectorstore(immigration_laws_collection, immigration_dir, db_path)
land_property_law_vector = init_vectorstore(land_property_laws_collection, land_property_dir, db_path)
tax_law_vector = init_vectorstore(tax_laws_collection, tax_dir, db_path)





#---------------- Create vectorstore retriever tool(s) ----------------

# Treat general law vectorstore as a retriever
law_retriever = law_vector.as_retriever()

# Create retriever tool
law_retriever_tool = create_retriever_tool(
    retriever=law_retriever, 
    name="law_retriever_tool", 
    description="Search for information about general laws and legal processes. For any questions about general laws in Rwanda, you must use this tool!"
    )



# Treat general law vectorstore as a retriever
bank_law_retriever = bank_law_vector.as_retriever()

# Create retriever tool
bank_law_retriever_tool = create_retriever_tool(
    retriever=bank_law_retriever, 
    name="bank_law_retriever_tool", 
    description="Search for information about bank and financial laws. For any questions about bank and financial laws in Rwanda, you must use this tool!"
    )



# Treat general law vectorstore as a retriever
company_law_retriever = company_law_vector.as_retriever()

# Create retriever tool
company_law_retriever_tool = create_retriever_tool(
    retriever=company_law_retriever, 
    name="company_law_retriever_tool", 
    description="Search for information about company laws. For any questions about company laws in Rwanda, you must use this tool!"
    )


# Treat general law vectorstore as a retriever
electronic_law_retriever = electronic_law_vector.as_retriever()

# Create retriever tool
electronic_law_retriever_tool = create_retriever_tool(
    retriever=electronic_law_retriever, 
    name="electronic_law_retriever_tool", 
    description="Search for information about electronic laws. For any questions about electronic laws in Rwanda, you must use this tool!"
    )



# Treat general law vectorstore as a retriever
immigration_law_retriever = immigration_law_vector.as_retriever()

# Create retriever tool
immigration_law_retriever_tool = create_retriever_tool(
    retriever=immigration_law_retriever, 
    name="immigration_law_retriever_tool", 
    description="Search for information about immigration and emigration. For any questions about immigration and emigration in Rwanda, you must use this tool!"
    )


# Treat general law vectorstore as a retriever
land_property_law_retriever = land_property_law_vector.as_retriever()

# Create retriever tool
land_property_law_retriever_tool = create_retriever_tool(
    retriever=land_property_law_retriever, 
    name="land_property_law_retriever_tool", 
    description="Search for information about land and property laws. For any questions about land and property laws in Rwanda, you must use this tool!"
    )


# Treat general law vectorstore as a retriever
tax_law_retriever = tax_law_vector.as_retriever()

# Create retriever tool
tax_law_retriever_tool = create_retriever_tool(
    retriever=tax_law_retriever, 
    name="tax_law_retriever_tool", 
    description="Search for information about tax laws. For any questions about tax laws in Rwanda, you must use this tool!"
    )





#---------------- Create Tavily Internet Search tool ----------------

# Define tavily api key as an environment variable
os.environ["TAVILY_API_KEY"] = "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"

# Initialize tavily search object
tavily_search = TavilySearchResults()





#---------------- Custom tools will be created here (if any needed) ----------------





#---------------- Define system prompt for the Chat Agent to know the tools to use ----------------

# Create a chat prompt template from a variety of message formats
prompt = ChatPromptTemplate.from_messages(
    messages=[
        (
            "system",

            '''
            You are an AI assistant. Your primary responsibility is to provide accurate and relevant information about Rwandan laws and processes, as well as general information, using the tools available to you.

            **Tool Usage**:
            
            - For questions about **general laws and legal processes** in Rwanda, use the `law_retriever_tool`. Always invoke this tool for any query related to laws in Rwanda, even when using other tools.
            
            - For questions about **bank and financial institution laws**, use the `bank_law_retriever_tool`. Always invoke this tool alongside the `law_retriever_tool` for these queries.
            
            - For questions about **company laws**, use the `company_law_retriever_tool`. Always invoke this tool alongside the `law_retriever_tool` for these queries.
            
            - For questions about **electronic laws**, use the `electronic_law_retriever_tool`. Always invoke this tool alongside the `law_retriever_tool` for these queries.
            
            - For questions about **immigration and emigration laws**, use the `immigration_law_retriever_tool`. Always invoke this tool alongside the `law_retriever_tool` for these queries.
            
            - For questions about **land and property laws**, use the `land_property_law_retriever_tool`. Always invoke this tool alongside the `law_retriever_tool` for these queries.
            
            - For questions about **tax laws**, use the `tax_law_retriever_tool`. Always invoke this tool alongside the `law_retriever_tool` for these queries.

            - For any other **general information**, use the `tavily_search` tool.

            **Behavioral Instructions**:
            
            - When the query is about a specific law, understand the query to determine the most appropriate tool to handle it. If the query spans multiple law categories, use all relevant tools together.
            
            - If no specific tool applies, or if the query is not about Rwandan laws, default to the `tavily_search` tool.

            **Language Handling**:
            
            - Always respond in **English** by default unless otherwise instructed.
            
            - If the query is sent in a specific language or the user instructs you to respond in a different language, respond in that language. Continue responding in the specified language until the user:
              1. Sends a query in another language, in which case switch to that language.
              2. Instructs you to change the language explicitly.

            - If the language in a query is ambiguous, respond in English and seek clarification.

            **General Principles**:
            
            - Always use the most appropriate tool(s) for the query.
            
            - Provide clear, accurate, and contextually relevant responses.
            
            - When required, invoke multiple tools simultaneously to ensure comprehensive information retrieval.
            '''
        ),

        MessagesPlaceholder(variable_name="chat_history"),

        ("user", "{input}"),

        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)




#---------------- Set Chart Starters ----------------

# @cl.set_starters
# async def set_starters():
#     return [
#         cl.Starter(
#             label="Business Registration Help",
#             message="Chat message to send here",
#             icon="/public/help-center.svg",
#             ),

#         cl.Starter(
#             label="Other type of query",
#             message="Chat message to send here",
#             icon="/public/maintenance.svg",
#             ),
#         cl.Starter(
#             label="Other type of query",
#             message="Chat message to send here",
#             icon="/public/error.svg",
#             ),
#         cl.Starter(
#             label="Other type of query",
#             message="Chat message to send here",
#             icon="/public/idea.svg",
#             )
#         ]



#---------------- Define what happens on chat start ----------------

@cl.on_chat_start
def setup_chain():

    # Define Base LLM to use
    llm = ChatOpenAI(
        openai_api_key="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",
        model="gpt-3.5-turbo"
    )

    # Define list of all pre-defined tools to use during chat
    tools = [law_retriever_tool, bank_law_retriever_tool, company_law_retriever_tool, electronic_law_retriever_tool, immigration_law_retriever_tool, land_property_law_retriever_tool, tax_law_retriever_tool, tavily_search]


    # Bind all the tools together to the llm chat model
    llm_with_tools = llm.bind_tools(tools)

    # Define the chat agent
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: x["chat_history"]
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )


    # Define the agent executer that uses the binded tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Set chainlit chat user session
    cl.user_session.set("llm_chain", agent_executor)





#---------------- Define what happens when a message is sent in the chat ----------------

# Container to store chat history
chat_history = []

city = ""
country = ""
results = ""
resultsDone = False

# Define the function to handle messages

@cl.on_message
async def handle_message(message: cl.Message):
    global city, country, results, resultsDone


    # convert the user message to all lower case
    user_message = message.content.lower()

    # Define an llm message chain for continuous conversation
    llm_chain = cl.user_session.get("llm_chain")

    # Store result from llm based on query sent from usermessage
    result = llm_chain.invoke(
        {
            "input": user_message,
            "chat_history": chat_history
        }
    )

    # append to chat history
    chat_history.extend(
        [
            HumanMessage(content=user_message),
            AIMessage(content=result["output"])
        ]
    )

    if resultsDone == False:  # not yet done, keep going around
        await cl.Message(result['output']).send()
    else:
        # send the add request to the UI
          
        fn = cl.CopilotFunction(name="formfill", args={"fieldA": city, "fieldB": country, "fieldC": result['output']})
        resultsDone = False
        res = await fn.acall()
        await cl.Message(content="Form info sent").send()