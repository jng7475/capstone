import os

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

from flask import Flask, request, jsonify
import constants
app = Flask(__name__)

chat_history = None
chain = None

@app.route('/', methods=['GET'])
def handle_request():
    global chat_history
    global chain
    input = str(request.args.get('input'))

    if input == 'None':
        return jsonify({'answer': 'I am here to assist with car-related questions and information. How can I assist you with your car today?'})
    else:
        # print(input, chat_history)
        result = chain({"question": input})
        # result = chain({'question': input, 'chat_history': chat_history})
        # chat_history.append((input, result['answer']))
        return jsonify({'answer': result['answer']})

@app.before_request 
def init():
    global chat_history
    global chain
    # api_key = os.getenv("OPENAI_API_KEY")
    api_key = constants.APIKEY
    os.environ["OPENAI_API_KEY"] = api_key

    # Enable to save to disk & reuse the model (for repeated queries on the same data)
    PERSIST = False

    query = None
    initial_prompt = "You're now a virtual assistant for a car. You will use your own knowledge about cars and services as well as the training data that I provided. Try to help the user with as much detail as possible. When asked a question, can you make shorter responses and provide step by step instruction. Only tell the user to contact the user's car's brand support team when you cannot find anything related to the issue from your own knowledge and training data. Don't tell them to look up information from the provided car manual, but briefly summarize to them using bullet points. Be nice and polite. Start by asking what the user needs help with."
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(base_dir, 'data')
    if PERSIST and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = FAISS(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
        # loader = DirectoryLoader("data/")
        loader = DirectoryLoader(data_dir_path)
        if PERSIST:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(),
        memory=memory,
    )
    result = chain({"question": initial_prompt})
    # chat_history = []
    # result = chain({"question": initial_prompt, "chat_history": chat_history})
    # chat_history.append((initial_prompt, 'I am here to assist with Toyota-related questions and information. How can I assist you with Toyota today?'))
    # print("Initial prompt: ", initial_prompt)

if __name__ == '__main__':
    init()
    app.run(debug=False, port=8000)

