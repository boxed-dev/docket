from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for your app
os.environ["OPENAI_API_KEY"] = ""

response_history = []
uploaded_file_path = None
llm = None
docsearch = None
index_creator = None
@app.route('/uploads', methods=['POST'])
def upload():
    global uploaded_file_path,llm,docsearch,index_creator
    try:
        file = request.files['file']
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        uploaded_file_path = file_path
        if uploaded_file_path is None:
            return jsonify({'error': 'No file uploaded'}), 400

        loader = CSVLoader(file_path=uploaded_file_path, encoding='utf-8')

        index_creator = VectorstoreIndexCreator()
        docsearch = index_creator.from_loaders([loader])

        # Use ChatOpenAI instead of OpenAIChat
        llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-0613')
        os.remove(file_path)
        return jsonify({'message': 'File uploaded successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_query', methods=['POST'])
def process_query():
    global uploaded_file_path,llm,index_creator,docsearch
    try:
        data = request.get_json()
        query = data['query']
        chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")

        response = chain({"question": query})
        result = response['result']

        response_history.append({'query': query, 'response': result})
        print(result)
        return jsonify({'result': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)