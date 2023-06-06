from langchain.document_loaders import PyPDFLoader,PlaywrightURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain import FAISS
import os
import pickle
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

urls =["https://jobs.apple.com/en-us/details/114438152/us-operations-expert?team=APPST"]

url_loader = PlaywrightURLLoader(urls = urls)
pdf_loader = PyPDFLoader("resume.pdf")

loaders=(pdf_loader,url_loader)

documents = []
for loader in loaders:
    documents.extend(loader.load())

text_splitter = CharacterTextSplitter(separator='\n',
                                      chunk_size=2000,
                                      chunk_overlap=200)


docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorStore_openAI = FAISS.from_documents(docs, embeddings)

with open("multiple_file_embeddings.pkl", "wb") as f:
    pickle.dump(vectorStore_openAI, f)

