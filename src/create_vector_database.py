from annoy import AnnoyIndex
import openai
import numpy as np
import json
EMBEDDING_DTYPE = np.float32
import os
import azure_keys
import src.azure_client as azure_client

openai.api_type = azure_keys.openai_api_type
openai.api_base = azure_keys.openai_api_base
openai.api_version = azure_keys.openai_api_version
openai.api_key = azure_keys.openai_api_key

f = 1536
t = AnnoyIndex(f, 'angular')
z = AnnoyIndex(f, 'angular')
idx = 0
idx_z = 0
text_dict = {}
text_dict_summary = {}
for i in range(2001, 2024):
#for i in [2001]:
    with open(f'../queens_speeches/speeches/{i}.txt', encoding='utf-8') as f:
        total_text = f"År {i}"
        try:
            for line_idx, line in enumerate(f.readlines()):
                total_text += '\n' + line
                line = line.strip()
                if len(line)>0:
                    line = f'År: {i}\n Linjenummer: {line_idx} \n' +  line
                    client = azure_client.initialize_client()
                    embedding_raw = client.embeddings.create(input=line, model="text-embedding-ada-002")
                    embedding = np.array(embedding_raw.data[0].embedding, dtype=EMBEDDING_DTYPE)
                    t.add_item(idx, embedding)
                    text_dict[idx] = line
                    idx += 1
        except Exception as exc:
            print(f'{f=}')
            raise exc
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

    #answer = raw_answer["choices"][0]["message"]["content"]
    answer = raw_answer
    # for summary_idx, summary_text in enumerate(answer.split('.')):
    #     summary_text = f'År: {i}\n Linjenummer: {summary_idx} \n' +  summary_text

    #     client = azure_client.initialize_client()
    #     embedding_raw = client.embeddings.create(input=summary_text, model="text-embedding-ada-002")
    #     embedding = np.array(embedding_raw.data[0].embedding, dtype=EMBEDDING_DTYPE)
    #     z.add_item(idx_z, embedding)
    #     text_dict_summary[idx_z] = summary_text
    #     idx_z += 1
    # print(i)

t.build(10) # 10 trees
t.save('queen_speeches.ann')
# z.build(10)
# z.save('queen_speeches_summaries.ann')

with open('processed_speeches.json', 'w') as fp:
    json.dump(text_dict, fp)
# with open('processed_speeches_summaries.json', 'w') as fp:
#     json.dump(text_dict_summary, fp)