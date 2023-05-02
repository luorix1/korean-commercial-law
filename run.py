from semantic_search import SemanticSearch
import openai
import os
import gradio as gr
from utils import pdf_to_text, text_to_chunks, generate_text, download_pdf

from dotenv import load_dotenv


recommender = SemanticSearch()

def load_recommender(path, start_page=1):
    global recommender
    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    recommender.fit(chunks)
    return "Corpus Loaded."
def generate_answer(question):
    topn_chunks = recommender(question)
    prompt = ""
    prompt += "search results:\n\n"
    for c in topn_chunks:
        prompt += c + "\n\n"

    prompt += (
        "Instructions: Compose a comprehensive reply to the query using the search results given. "
        "Cite each reference using [number] notation (every result has this number at the beginning). "
        "Citation should be done at the end of each sentence. If the search results mention multiple subjects "
        "with the same name, create separate answers for each. Only include information found in the results and "
        "don't add any additional information. Make sure the answer is correct and don't output false content. "
        "If the text does not relate to the query, simply state 'Found Nothing'. Ignore outlier "
        "search results which has nothing to do with the question. Only answer what is asked. The "
        "answer should be short and concise.\n\nQuery: {question}\nAnswer: "
    )

    prompt += f"Query: {question}\nAnswer:"
    answer = generate_text(prompt)
    return answer


def question_answer(url, file, question, api_key):
    openai.api_key = api_key

    if url.strip() == "" and file == None:
        return "[ERROR]: Both URL and PDF is empty. Provide atleast one."

    if url.strip() != "" and file != None:
        return "[ERROR]: Both URL and PDF is provided. Please provide only one (eiter URL or PDF)."

    if url.strip() != "":
        glob_url = url
        download_pdf(glob_url, "corpus.pdf")
        load_recommender("corpus.pdf")

    else:
        old_file_name = file.name
        file_name = file.name
        file_name = file_name[:-12] + file_name[-4:]
        os.rename(old_file_name, file_name)
        load_recommender(file_name)

    if question.strip() == "":
        return "[ERROR]: Question field is empty"

    return generate_answer(question)


title = 'CommercialLawGPT'
description = "CommercialLawGPT allows you to ask questions about South Korean commercial law. This app uses GPT-3 to generate answers based on the book's information. BookGPT has ability to add reference to the specific page number from where the information was found. This adds credibility to the answers generated also helps you locate the relevant information in the book."

with gr.Blocks() as demo:

    gr.Markdown(f'<center><h1>{title}</h1></center>')
    gr.Markdown(description)
    gr.Markdown("Thank you for all the support this space has received! Unfortunately, my OpenAI $18 grant has been exhausted, so you'll need to enter your own OpenAI API Key to use the app. Sorry for inconvenience :-(.")

    with gr.Row():
        
        with gr.Group():
            url = gr.Textbox(label='URL')
            gr.Markdown("<center><h6>or<h6></center>")
            file = gr.File(label='PDF', file_types=['.pdf'])
            question = gr.Textbox(label='question')
            api_key = gr.Textbox(label='OpenAI API Key')
            btn = gr.Button(value='Submit')
            btn.style(full_width=True)

        with gr.Group():
            answer = gr.Textbox(label='answer')

        btn.click(question_answer, inputs=[url, file, question, api_key], outputs=[answer])

demo.launch()