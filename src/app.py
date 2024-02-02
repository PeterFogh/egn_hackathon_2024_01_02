import json

import numpy as np
import openai
import streamlit as st
from annoy import AnnoyIndex
from streamlit_chat import message
import azure_client as azure_client


def _submit():
    st.session_state['question'] = st.session_state.input_field
    st.session_state.input_field = ''


def app():
    # Build Streamlit App
    # Setup azure openai

    st.title("Team 1")
    st.title("Kapacity LLM Hackathon")
    f = 1536
    EMBEDDING_DTYPE = np.float32

    annoy_level_line = AnnoyIndex(f, 'angular')
    annoy_level_line.load('queen_speeches.ann')
    annoy_level_summary = AnnoyIndex(f, 'angular')
    annoy_level_summary.load('queen_speeches_summaries.ann')
    annoy_level_file = AnnoyIndex(f, 'angular')
    annoy_level_file.load('queen_speeches_files.ann')

    #a = AnnoyIndex(f, 'angular')
    #a.load('queen_speeches_summaries.ann')
    with open('processed_speeches.json') as fp:
        text_dict_line = json.load(fp)
    with open('processed_speeches_summaries.json') as fp:
        text_dict_summary = json.load(fp)
    with open('processed_speeches_files.json') as fp:
        text_dict_file = json.load(fp)

    if 'question' not in st.session_state:
        st.session_state['question'] = ''

    placeholder = st.empty()

    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = [{"message": "Hi! How may I help you today?", "is_user": False}]
        with placeholder.container():
            message(st.session_state['conversation'][0]["message"], is_user=False, key="start")

    st.text_input("you:", key="input_field", on_change=_submit)

    count = 0
    if st.session_state['question']:
        with placeholder.container():
            for message_ in st.session_state['conversation']:
                message(message_["message"], is_user=message_["is_user"], key=count)
                count += 1

        with st.spinner(f"Generate answer for question: {st.session_state['question']}"):

            client = azure_client.initialize_client()
            embedding_raw = client.embeddings.create(input=st.session_state['question'], model="text-embedding-ada-002")
            embedding = np.array(embedding_raw.data[0].embedding, dtype=EMBEDDING_DTYPE)
            vectors_line = annoy_level_line.get_nns_by_vector(embedding, 5, search_k=-1, include_distances=True)
            vectors_summaries = annoy_level_summary.get_nns_by_vector(embedding, 10, search_k=-1, include_distances=True)
            vectors_file = annoy_level_summary.get_nns_by_vector(embedding, 3, search_k=-1, include_distances=True)

            texts_line = [text_dict_line[str(i)] for i in vectors_line[0]]
            texts_summary = [text_dict_summary[str(i)] for i in vectors_summaries[0]]
            texts_file = [text_dict_line[str(i)] for i in vectors_file[0]]

            #summaries = [text_summaries_dict[str(i)] for i in vectors_summaries[0]]
                     #You are answering questions to the best of your ability.
            selected_conversation_hist = [
                {"role": "system",
                 "content": f"""


                    Søg efter navngivne enheder i teksten og den sammenhæng, den er i, samt året. Fokuser især på titler og personer i den kongelige familie.
                    Hold svaret kort, vær hjælpsom og høflig. Vi diskuterer dronningens nytårstaler,
                    og jeg vil besvare spørgsmål detaljeret inden for konteksten af ​​hver specifikke forespørgsel,
                    med særlig opmærksomhed på medlemmer og titler i den kongelige familie.
                     """},
            ]
            #print(texts)
            modified_question = "Brugeren spurgte: \n" + \
                                st.session_state['question'] + \
                                '\n Du har nu følgende fra dronningens nytårstaler fra 2001-2023 at svare ud fra: \n' + \
                                '\n '.join(texts_line) + \
                                '\n Og disse følgende tekster fra opsummeringer: \n' + \
                                '\n '.join(texts_summary) + \
                                '\n Og denne tale: \n' + \
                                '\n '.join(texts_file) + \
                                '\n Besvar spørgsmålet.'
            # modified_question = "Brugeren spurgte: \n" + \
            #                     st.session_state['question'] + \
            #                     '\n Du har nu følgende fra dronningens nytårstaler fra 2001-2022 at svare ud fra: \n' + \
            #                     '\n '.join(texts) + \
            #                     '\n Og disse følgende tekster fra opsummeringer: \n' + \
            #                     '\n '.join(summaries) + \
            #                     '\n Besvar spørgsmålet.'
            print(modified_question)
            # Add question to conversation
            messages = selected_conversation_hist + [{"role": "user", "content": modified_question}]

            raw_answer = client.chat.completions.create(
                model="gpt-35-turbo",
                messages=messages
            )
            print(raw_answer)
            print(type(raw_answer))
            print(raw_answer.choices)
            answer = raw_answer.choices[0].message.content
            #answer = raw_answer

        # Save new question and answer in the session state
        st.session_state['conversation'].append({"message": st.session_state['question'], "is_user": True})
        st.session_state['conversation'].append({"message": answer, "is_user": False})

        # Show new question and answer
        with placeholder.container():
            for message_ in st.session_state['conversation']:
                message(message_["message"], is_user=message_["is_user"], key=count)
                count += 1

        # Reset question state
        st.session_state['question'] = ''


if __name__ == "__main__":
    app()
