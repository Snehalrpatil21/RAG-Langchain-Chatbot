from flask import Flask, request, render_template, redirect, url_for, session
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY')

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key

@app.route('/')
def index():
    response = session.pop('response', None)
    sources = session.pop('sources', None)
    return render_template('index.html', response=response, sources=sources)

@app.route('/query', methods=['POST'])
def query():
    query_text = request.form['query']
    
    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        session['response'] = "Unable to find matching results."
        session['sources'] = []
        return redirect(url_for('index'))

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI()
    response_text = model.predict(prompt)


    sources = [{"source": doc.metadata.get("source"), "page": doc.metadata.get("page"), "chunk_index": doc.metadata.get("start_index")} for doc, _score in results]
    
    session['response'] = response_text
    session['sources'] = sources
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)