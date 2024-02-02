from annoy import AnnoyIndex
import openai
import numpy as np
import json
EMBEDDING_DTYPE = np.float32
import os
import azure_client as azure_client

f = 1536
list_of_lines = AnnoyIndex(f, 'angular')
list_of_summaries = AnnoyIndex(f, 'angular')
list_of_files = AnnoyIndex(f, 'angular')
id_lines = 0
id_summaries = 0
id_files = 0
text_dict_lines = {}
text_dict_summary = {}
text_dict_files = {}

for file in range(2001, 2024):
#for i in [2001]:
    with open(f'../queens_speeches/speeches/{file}.txt', encoding='utf-8') as f:
        total_text = f"År {file}"
        for line_idx, line in enumerate(f.readlines()):
            total_text += '\n' + line
            line = line.strip()
            if len(line)>0:
                line = f'År: {file}\n Linjenummer: {line_idx} \n' +  line
                client = azure_client.initialize_client()
                embedding_raw = client.embeddings.create(input=line, model="text-embedding-ada-002")
                embedding = np.array(embedding_raw.data[0].embedding, dtype=EMBEDDING_DTYPE)
                list_of_lines.add_item(id_lines, embedding)
                text_dict_lines[id_lines] = line
                id_lines += 1
    selected_conversation_hist = [
        {"role": "system",
         "content": f"""
            Your job is to summarize the Danish Queens new years speeches.
            Please include information such as but not limited to
            themes, people, countries, facts, events and minorities.
            Please do so in Danish.
             """},
    ]
    modified_question = "Opsummer følgende tale: \n" + \
                        total_text
    # Add question to conversation
    messages = selected_conversation_hist + [{"role": "user", "content": modified_question}]

    client = azure_client.initialize_client()
    raw_answer = client.chat.completions.create(
        model="gpt-35-turbo",
        messages=messages
    )

    answer = raw_answer.choices[0].message.content
    for summary_idx, summary_text in enumerate(answer.split('.')):
        summary_text = f'År: {file}\n Linjenummer: {summary_idx} \n' +  summary_text

        client = azure_client.initialize_client()
        embedding_raw = client.embeddings.create(input=summary_text, model="text-embedding-ada-002")
        embedding = np.array(embedding_raw.data[0].embedding, dtype=EMBEDDING_DTYPE)
        list_of_summaries.add_item(id_summaries, embedding)
        text_dict_summary[id_summaries] = summary_text
        id_summaries += 1

    client = azure_client.initialize_client()
    embedding_raw = client.embeddings.create(input=total_text, model="text-embedding-ada-002")
    embedding = np.array(embedding_raw.data[0].embedding, dtype=EMBEDDING_DTYPE)
    list_of_files.add_item(id_files, embedding)
    text_dict_files[id_files] = total_text
    id_files += 1
    print(f'Processed file: {file}')

print('writing annoy files')
list_of_lines.build(10) # 10 trees
list_of_lines.save('queen_speeches.ann')
list_of_summaries.build(10)
list_of_summaries.save('queen_speeches_summaries.ann')
list_of_files.build(10)
list_of_files.save('queen_speeches_files.ann')


print('writing json files')
with open('processed_speeches.json', 'w') as fp:
    json.dump(text_dict_lines, fp)
with open('processed_speeches_summaries.json', 'w') as fp:
    json.dump(text_dict_summary, fp)
with open('processed_speeches_files.json', 'w') as fp:
    json.dump(text_dict_files, fp)