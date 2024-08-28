from langchain_openai import OpenAI 
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import youtube
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings()

parser= StrOutputParser()

def create_db_from_youtube_video_url ( video_url:str) -> faiss.FAISS:
    loader = youtube.YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = faiss.FAISS.from_documents(docs, embeddings)

    return db

def get_response_from_query(db, query, k=4):
    """
    text-davinci-003 can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model_name = "gpt-3.5-turbo-instruct")


    prompt = PromptTemplate(
            input_variables= ['question', 'docs'],
            template=""" You are a helpful assistant that can answer questions about youtube videos 
                        based on the video's transcript.
                        
                        Answer the following question: {question}
                        By searching the following video transcript: {docs}
                        
                        Only use the factual information from the transcript to answer the question.
                        
                        If you feel like you don't have enough information to answer the question, say "I don't know".
                        Your answers should be verbose and detailed.
                    """
    )

    chain = prompt | llm | parser

    response = chain.invoke({"question": query, "docs":docs_page_content})

    return response

if __name__ == "__main__":
    db= create_db_from_youtube_video_url("") 
    print(get_response_from_query(db, ""))

