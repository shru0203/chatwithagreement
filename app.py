

# import streamlit as st
# import time
# from dotenv import load_dotenv
# import os
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain_community.llms import HuggingFaceHub
# import base64


# # Function to set background image
# def set_png_as_page_bg(png_file):
#     bin_str = get_base64_of_bin_file(png_file)
#     page_bg_img = f'''
#     <style>
#     .stApp {{
#         background-image: url("data:image/png;base64,{bin_str}");
#         background-size: cover;
#     }}
#     </style>
#     '''
#     st.markdown(page_bg_img, unsafe_allow_html=True)


# # Function to convert image to base64
# def get_base64_of_bin_file(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()


# def update_chat_log(sender, text):
#     if 'chat_log' not in st.session_state:
#         st.session_state.chat_log = []
#     st.session_state.chat_log.append({"sender": sender, "text": text})


# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Ask PDF/Agreement")
#     st.header("Ask PDF/Agreement")
#     st.sidebar.header("Your uploaded PDFs")

#     # Set background image
#     set_png_as_page_bg('beige.jpg')

#     # Upload a file
#     pdf = st.sidebar.file_uploader("Upload PDF/Agreement", type="pdf")

#     # Update the progress bar as the file uploads
#     if pdf is not None:
#         my_bar = st.sidebar.progress(0)
#         for i in range(100):
#             my_bar.progress(i + 1)
#             time.sleep(0.1)

#     if pdf is not None:
#         pdf_reader = PdfReader(pdf)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()

#         # Split into chunks
#         text_splitter = CharacterTextSplitter(
#             separator="\n",
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len
#         )
#         chunks = text_splitter.split_text(text)

#         # Create embeddings
#         embeddings = HuggingFaceEmbeddings()

#         knowledge_base = FAISS.from_texts(chunks, embeddings)

#         user_question = st.text_input("Ask Question about PDF/Agreement:",value=" ")
#         if user_question:
#             docs = knowledge_base.similarity_search(user_question)
#             llm = HuggingFaceHub(repo_id="google/flan-t5-large",
#                                  model_kwargs={"temperature": 5, "max_length": 64},
#                                  huggingfacehub_api_token="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")
#             chain = load_qa_chain(llm, chain_type="stuff")
#             response = chain.run(input_documents=docs, question=user_question)

#             # Update chat history using the wrapper function
#             update_chat_log("You", user_question)
#             update_chat_log("Assistant", response)

#             # Display chat history
#             for message in st.session_state.chat_log:
#                 st.text(f"{message['sender']}: {message['text']}")

          



# if __name__ == '__main__':
#     main()






# import streamlit as st
# import time
# from dotenv import load_dotenv
# import os
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain_community.llms import HuggingFaceHub
# import base64

# # Function to set background image
# def set_png_as_page_bg(png_file):
#     bin_str = get_base64_of_bin_file(png_file)
#     page_bg_img = f'''
#     <style>
#     .stApp {{
#         background-image: url("data:image/png;base64,{bin_str}");
#         background-size: cover;
#     }}
#     </style>
#     '''
#     st.markdown(page_bg_img, unsafe_allow_html=True)

# # Function to convert image to base64
# def get_base64_of_bin_file(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# # Function to update chat log
# def update_chat_log(sender, text):
#     if 'chat_log' not in st.session_state:
#         st.session_state.chat_log = []
#     st.session_state.chat_log.append({"sender": sender, "text": text})

# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Ask PDF/Agreement")
#     st.header("Ask PDF/Agreement")
#     st.sidebar.header("Your uploaded PDFs")

#     # Set background image
#     set_png_as_page_bg('beige.jpg')

#     # Upload a file
#     pdf = st.sidebar.file_uploader("Upload PDF/Agreement", type="pdf")

#     # Update the progress bar as the file uploads
#     if pdf is not None:
#         my_bar = st.sidebar.progress(0)
#         for i in range(100):
#             my_bar.progress(i + 1)
#             time.sleep(0.1)

#     if pdf is not None:
#         pdf_reader = PdfReader(pdf)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()

#         # Split into chunks
#         text_splitter = CharacterTextSplitter(
#             separator="\n",
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len
#         )
#         chunks = text_splitter.split_text(text)

#         # Create embeddings
#         embeddings = HuggingFaceEmbeddings()
#         knowledge_base = FAISS.from_texts(chunks, embeddings)

#         # Text input for user question
#         user_question = st.text_input("Ask Question about PDF/Agreement:", value=" ")
        
#         if user_question.strip():  # Ensure user_question is not empty or just whitespace
#             docs = knowledge_base.similarity_search(user_question)
#             llm = HuggingFaceHub(repo_id="google/flan-t5-large",
#                                  model_kwargs={"temperature": 5, "max_length": 64},
#                                  huggingfacehub_api_token="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")
#             chain = load_qa_chain(llm, chain_type="stuff")
#             response = chain.run(input_documents=docs, question=user_question)

#             # Update chat history
#             update_chat_log("You", user_question)
#             update_chat_log("Assistant", response)

#             # Display chat history
#             for message in st.session_state.chat_log:
#                 if message['sender'] == 'Assistant':
#                     avatar = '🤓'  # Nerd emoji for assistant
#                 else:
#                     avatar = '👤'  # User emoji

#         # Display message with avatar
#                 with st.chat_message(message['sender'], avatar=avatar):
#                     st.write(message['text'])



# if __name__ == '__main__':
#     main()

import streamlit as st
import time
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub
import base64

# Function to set background image
def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Function to convert image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to update chat log
def update_chat_log(sender, text):
    if 'chat_log' not in st.session_state:
        st.session_state.chat_log = []
    st.session_state.chat_log.append({"sender": sender, "text": text})

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask PDF/Agreement")
    st.header("Ask PDF/Agreement")
    st.sidebar.header("Your uploaded PDFs")

    # Set background image
    set_png_as_page_bg('beige.jpg')

    # Upload a file
    pdf = st.sidebar.file_uploader("Upload PDF/Agreement", type="pdf")

    # Clear the chat_log and screen if a new PDF is uploaded
    if pdf is not None:
        st.session_state.pop('chat_log', None)
        st.empty()

    # Update the progress bar as the file uploads
    if pdf is not None:
        my_bar = st.sidebar.progress(0)
        for i in range(100):
            my_bar.progress(i + 1)
            time.sleep(0.1)

    if pdf is not None:
        # Read PDF content and process it
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # User input for question
        user_question = st.text_input("Ask Question about PDF/Agreement:", value=" ")

        if user_question.strip():  
            docs = knowledge_base.similarity_search(user_question)
            llm = HuggingFaceHub(repo_id="google/flan-t5-large",
                                 model_kwargs={"temperature": 5, "max_length": 64},
                                 huggingfacehub_api_token="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")
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            # Update chat history
            update_chat_log("You", user_question)
            update_chat_log("Assistant", response)

            # Display chat history
            for message in st.session_state.chat_log:
                avatar = '🤓' if message['sender'] == 'Assistant' else '👤'
                with st.chat_message(message['sender'], avatar=avatar):
                    st.write(message['text'])

if __name__ == '__main__':
    main()