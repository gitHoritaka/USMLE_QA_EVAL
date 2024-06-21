from langchain.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores.faiss import FAISS
import configparser

def read_config():
    config = configparser.ConfigParser()
    config.read('config.cfg')
    return config


def setup_local_rag(config):
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large"
    )

    # ディレクトリのパスを指定
    faiss_directory_path = config["DEFAULT"]["FAISS_DIRECTORY_PATH"]
    faiss_db = FAISS.load_local(faiss_directory_path, embeddings=embedding_model,allow_dangerous_deserialization=True)
    # Retrievers
    retriever = faiss_db.as_retriever()
    return retriever


