##### **Project Title: MultiPDFChat - Document Interaction Simplified**



**Goal**

To build a smart chatbot that allows users to ask questions across multiple PDF documents an receive accurate, real-time answers using RAG.



**How the RAG chain works?**

* User uploads PDFs

  * Multiple documents are uploaded and stored for processing
* Document Chunking

  * Each PDF is split into manageable text chunks
* Embedding generation

  * Use an embedding model (e.g., text-embedding-ada-002) to convert chunks into vectors
* Indexing in Vector DB

  * Store these embeddings in a vector DB (e.g., FAISS, ChromaDB)
* User Enters a Query

  * User types a question like: "What is the refund policy mentioned in any PDF"
* Query Embedding

  * The question is embedded using the same embedding model
* Similarity Search

  * Perform a vector search to retrieve the most relevant document chunks
* Query + Context Augmentation

  * Combine user query with top-matching chunks
* LLM Generation

  * Pass the combined input to an LLM (e.g., GPT-4 or Gemini) to generate the final answer
* Response Delivery

  * Display the result to the user in a chat-like interface



**Datasets (PDFs)**

* Attention is all You Need research paper
* You Only Look Once research paper



**Step-1:- Install required libraries**

Install core libraries for:

* PDF reading (PyPDF2)
* LangChain framework
* Google Gemini for embeddings and chat
* FAISS for vector similarity search

|**Library**|**Purpose**|
|-|-|
|PyPDF2|Read PDF Files|
|langchain|Build the RAG pipeline|
|langchain\_community|FAISS integration|
|langchain\_google\_genai|Connect LC with Gemini|
|FAISS|Vector similarity search|
|google-generativeai|Google Generative API|
|Streamlit|To create a web app and make accessible to multiple users|



**Step-2:- Import Libraries**

Load tools for:

* Reading PDFs
* Splitting text
* Generating embeddings
* Chatting  with Gemini
* Building RAG chains



**Step-3:- Set Gemini API Key**

Authorizes access to Google Gemini services using API key



**Step-4:- Extract Text from PDFs**

Roads multiple PDFs and returns their combined text content



**Step-5:- Split Text into Chunks**

Splits long text into overlapping chunks to preserve context (important for QA)



**Step-6:- Create Embeddings \& Store in FAISS (local folder)**

Converts chunks to embeddings using Gemini

Stores them in FAISS for similarity search 



**Step-7:- Create Conversational Chain (QA logic)**

Defines how to answer questions using:

* Retrieved documents
* Gemini LLM with custom prompt



**Step-8:- Handle User Questions**

* Retrieves relevant chunks from FAISS using the user's question
* Sends it to Gemini with the prompt template
* Prints the AI-generated answer



**Step-9:-Main Orchestration Function**

main() is used as the starting/controller function of Python application. It is not mandatory in Python, but it makes a larger program easier to organize and control. Functions help break a program into smaller, manageable tasks.

main() is the entry point and controller of our RAG application. It decides what should happen first, what should happen next, calls the appropriate functions to execute each stage.

Controls the full pipeline:

* Loads and processes PDF
* Creates embeddings and stores them
* Repeatedly accepts user questions and answers them using RAG







###### **Streamlit (app.py)**

Add your API key in .streamlit/secrets.toml 

run the application as \[python -m streamlit run app.py]

After running the applications, upload the PDFs you want explanation.





###### **Future Updates**

* Can add user-based logins and sessions

