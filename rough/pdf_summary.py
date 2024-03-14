#load document
from langchain_community.llms import Ollama
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
llm = Ollama(model='llama2')
#loader = PyPDFLoader("book.pdf")

#documents = loader.load()

def summarize_pdf(pdf_file_path, custom_prompt=""):
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load_and_split()
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.invoke(docs)
    print(summary)
    return summary

summarize_pdf('Demo_Patient_Report.pdf', custom_prompt="summarize pdf")