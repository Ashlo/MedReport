#load document
import gradio as gr
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
    summary = format_summary_to_bullet_points(summary)
    return summary

def format_summary_to_bullet_points(summary_data):
    # Extract the 'output_text' field from the summary data
    output_text = summary_data.get('output_text', '')

    # Split the output text into sentences
    sentences = output_text.split('. ')

    # Format each sentence as a bullet point
    bullet_points = ["- " + sentence.strip() for sentence in sentences if sentence]

    # Combine the bullet points into a single string
    formatted_summary = "\n".join(bullet_points)

    return formatted_summary

def main():
    input_pdf_path = gr.File(label="Upload PDF")
    output_summary=  gr.Textbox(label="Summary")

    iface = gr.Interface(
        fn=summarize_pdf,
        inputs=input_pdf_path,
        outputs=output_summary,
        title="MedReport",
        description="Enter path to MedReport"
    )

    iface.launch()


if __name__=="__main__":
    main()