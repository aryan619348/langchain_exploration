from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain import FAISS
import os
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

loader = PyPDFLoader("resume.pdf")
data = loader.load_and_split()

text_splitter = CharacterTextSplitter(separator='\n',
                                      chunk_size=2000,
                                      chunk_overlap=200)


docs = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings()
vectorStore_openAI = FAISS.from_documents(docs, embeddings)

vectorStore_openAI.save_local("pdf_embeddings")