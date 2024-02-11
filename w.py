import streamlit as st
import re
from bs4 import BeautifulSoup
from bs4.element import Comment
import requests
import re
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering, pipeline
import torch
from scipy.spatial.distance import cosine
import numpy as np

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    comm = re.compile("<!--|-->")
    soup = BeautifulSoup(comm.sub("",body.text),'lxml')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', truncation=True, max_model_input_size = 512)

model = AutoModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True,
                                  )

model_name = "deepset/roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {
    'question': 'What is the the intended use/patient and the effect of the therapeutic service/device?',
    'context': 'The webpage should include the indication for use of the therapeutic services/devices specifying the disease, symptoms or condition and the effects whether it is the diagnoses, treats, prevents, cures or mitigates.'
}

QA_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
QA_tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_bert_embeddings(text):
    # Add special tokens adds [CLS] and [SEP] tokens
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Predict hidden states features for each layer
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]

    # Use the average of the last 2 layers' hidden states as the sentence embedding
    token_vecs = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding

st.title("ReguTx Chatbot")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

#start the conversation off with a message from the assistant
st.session_state.messages.append(
    {
        "role": "assistant",
        "content": "Hello! I'm a chatbot trained by ReguTx to inform you about the reliability of therapeutic/medical services. Let me know of any that is on your mind! (!help for manual)",
    }
)

# Display chat messages from history on app rerun
def display_last_message():
    message = st.session_state.messages[len(st.session_state.messages)-1]

    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    
display_last_message()

url_pattern = "^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$"

# Accept user input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    display_last_message()

    with st.chat_message("assistant"):
        if (prompt == "!help"):
            response = """Function 1: Input URL for NLP analysis of therapeutic service!; Function 2: !display for Display database of international healthcare services documented by Crunchbase"""
        
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)
        elif (re.match(url_pattern, prompt)):
            response = "This is URL option: " + prompt
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)
            try:
                html = requests.get(prompt)
            except requests.exceptions.ConnectionError:
                response = "Connection Error to URL"
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)
            
            if html.status_code != 200:
                response = "Connection Error to URL"
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)
            else:
                text = text_from_html(html)
                # text = np.reshape(text, (1, 512))
                query = "This therapeutic service will diagnose, treat, prevent, cure or mitigate a disease, illness or sickness."

                sentences = text.split('. ')

                query_embedding = get_bert_embeddings(query).numpy()

                similarities = []
                for sentence in sentences:
                    sentence_embedding = get_bert_embeddings(sentence).numpy()
                    similarity = 1 - cosine(query_embedding, sentence_embedding)
                    similarities.append(similarity)

                most_similar_sentence_index = np.argmax(similarities)

                if (similarities[most_similar_sentence_index] < 0.35):
                    response = "Probably not a therapeutic/healthcare service. Please try again"
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.markdown(response)
                elif (similarities[most_similar_sentence_index] < 0.65):
                    response = "Treatment/therapeutic service is not evidence based."
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.markdown(response)
                else:
                    response = "Treatment/therapeutic service is evidence based. "
                    res = nlp(QA_input)
                    print(res['answer'])
                    st.session_state.messages.append({"role": "assistant", "content": response+res['answer']})
                    st.markdown(response+res['answer'])


        elif (prompt == "!display"):
            response = "Displaying database:"
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)
        else:
            response = "Not relevent! Ask GPT"

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)
