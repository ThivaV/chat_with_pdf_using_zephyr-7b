import os
import tempfile

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import CTransformers
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from streamlit_extras.add_vertical_space import add_vertical_space


@st.cache_resource(ttl="1h")
def get_retriever(pdf_files):
    """get retriever"""

    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for pdf_file in pdf_files:
        temp_pdf_file_path = os.path.join(temp_dir.name, pdf_file.name)

        with open(temp_pdf_file_path, "wb") as f:
            f.write(pdf_file.getvalue())

        loader = PyPDFLoader(temp_pdf_file_path)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1500, chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)

    # get huggingface token from env secret
    HF_TOKEN = os.environ.get("HF_TOKEN")

    # embeddings
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN,
        model_name="BAAI/bge-base-en-v1.5",
    )

    # retrieve k
    k = 5

    # vector retriever
    vector_store = Chroma.from_documents(chunks, embeddings)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # semantic retriever
    semantic_retriever = BM25Retriever.from_documents(chunks)
    semantic_retriever.k = k

    # ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, semantic_retriever], weights=[0.5, 0.5]
    )

    return ensemble_retriever


@st.cache_resource(ttl="1h")
def initialize_llm(_retriever):
    """initialize llm"""

    # load llm model
    model_type = "mistral"
    model_id = "TheBloke/zephyr-7B-beta-GGUF"
    model_file = "zephyr-7b-beta.Q4_K_S.gguf"

    config = {
        "max_new_tokens": 2048,
        "repetition_penalty": 1.1,
        "temperature": 1,
        "top_k": 50,
        "top_p": 0.9,
        "stream": True,
        "context_length": 4096,
        "gpu_layers": 0,
        "threads": int(os.cpu_count()),
    }

    llm = CTransformers(
        model=model_id,
        model_file=model_file,
        model_type=model_type,
        config=config,
        lib="avx2",
    )

    chat_history = StreamlitChatMessageHistory()

    # init chat history memory
    memory = ConversationBufferMemory(
        memory_key="chat_history", chat_memory=chat_history, return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=_retriever, memory=memory, verbose=False
    )

    return chain, chat_history


def main():
    """main func"""

    st.set_page_config(
        page_title="Talk to PDF using Zephyr-7B-Beta",
        page_icon="📰",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.header("Talk to PDF files 📰", divider="rainbow")
    st.subheader(
        "Enjoy :red[talking] with :green[PDF] files using :sunglasses: Zephyr-7B-Beta"
    )
    st.markdown(
        """
            * Used the [zephyr-7b-beta.Q4_K_S.gguf](https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/blob/main/zephyr-7b-alpha.Q4_K_S.gguf) quantised 
            version of [Zephyr-7B Beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) model 
            from the [TheBloke/zephyr-7B-beta-GGUF](https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF) repositry.
            ___
        """
    )

    st.sidebar.title("Talk to PDF 📰")
    st.sidebar.markdown(
        "[Checkout the repository](https://github.com/ThivaV/chat_with_pdf_using_zephyr-7b)"
    )
    st.sidebar.markdown(
        """
            ### This is a LLM powered chatbot, built using:
                
            * [Streamlit](https://streamlit.io)
            * [LangChain](https://python.langchain.com/)
            * [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)
            * [TheBloke/zephyr-7B-beta-GGUF](https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF)
            * [CTransformers](https://github.com/marella/ctransformers)
            * [Embeddings](https://huggingface.co/BAAI/bge-base-en-v1.5)
            * [Chroma](https://docs.trychroma.com/?lang=py)
            ___
            """
    )

    add_vertical_space(2)

    upload_pdf_files = st.sidebar.file_uploader(
        "Upload a pdf files 📤", type="pdf", accept_multiple_files=True
    )

    if not upload_pdf_files:
        st.info("👈 :red[Please upload pdf files] ⛔")
        st.stop()

    retriever = get_retriever(upload_pdf_files)

    chain, chat_history = initialize_llm(retriever)

    # load previous chat history
    # re-draw the chat history in the chat window
    for message in chat_history.messages:
        st.chat_message(message.type).write(message.content)

    if prompt := st.chat_input("Ask questions"):
        with st.chat_message("human"):
            st.markdown(prompt)

        response = chain.invoke(prompt)

        with st.chat_message("ai"):
            st.write(response["answer"])


if __name__ == "__main__":
    # init main func
    main()
