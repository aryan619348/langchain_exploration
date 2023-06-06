from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
from langchain.document_loaders import SeleniumURLLoader
from langchain.document_loaders import PlaywrightURLLoader
from langchain.document_loaders import WebBaseLoader,UnstructuredURLLoader

import pickle
from dotenv import load_dotenv
from langchain import FAISS
import os
import nest_asyncio
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
nest_asyncio.apply()
#create embeddings for a website:

urls =["https://jobs.apple.com/en-us/details/114438152/us-operations-expert?team=APPST"]

loader = PlaywrightURLLoader(urls = urls)

data = loader.load()


text_splitter = CharacterTextSplitter(separator='\n',
                                      chunk_size=2000,
                                      chunk_overlap=200)


docs = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings()
vectorStore_openAI = FAISS.from_documents(docs, embeddings)

with open("jobsite_embeddings.pkl", "wb") as f:
    pickle.dump(vectorStore_openAI, f)
