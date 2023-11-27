
import pinecone
from openai import OpenAI
from dotenv import dotenv_values

#Loading Credentials
env_name = "credentials.env"
config = dotenv_values(env_name)
client = OpenAI(api_key= config["openai_api"])


#Connection
index_name = config["index_name"]
# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key =  config["Pinecone_api_key"],
    environment =  config["Pinecone_environment"]
)
index = pinecone.Index(index_name)

#Vector Search
def Vector_search(query):
  Rag_data = ""
  xq = client.embeddings.create(input=query,model="text-embedding-ada-002")
  res = index.query([xq.data[0].embedding], top_k=2, include_metadata=True)
  for match in res['matches']:
      if match['score'] < 0.80:
        continue 
      Rag_data += match['metadata']['text']
  return Rag_data 

#GPT Completion
def GPT_completion_with_vector_search(prompt, rag):
    DEFAULT_SYSTEM_PROMPT = '''You are a helpful, respectful and honest INTP-T AI Assistant named Gathnex AI. You are talking to a human User.
    Always answer as helpfully and logically as possible, while being safe. Your answers should not include any harmful, political, religious, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    You also have access to RAG vectore database access which has Indian Law data. Be careful when giving response, sometime irrelevent Rag content will be there so give response effectivly to user based on the prompt.
    You can speak fluently in English.
    Note: Sometimes the Context is not relevant to Question, so give Answer according to that based on sutiation.
    '''
    response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {f"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {f"role": "user", "content": rag +", Prompt: "+ prompt},
    ]
    )

    return response.choices[0].message.content
