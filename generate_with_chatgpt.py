#%%
import os
import openai
import pandas as pd

openai.organization = "write_yours"
openai.api_key = "write_yours"

# %%
df = pd.read_csv(f'.dataset_balanced.csv')
df.label = df.label.apply(lambda x: 1 if x == "positive" else 0)
df = df[:29760].sample(frac=1).reset_index(drop=True)

#%%
df_similar = pd.DataFrame(columns=['text','label'])

df_to_generate = df[:200]
for i in df_to_generate.iterrows():
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    messages=[{"role": "user",
    "content":  f"Give me 10 similar short sentences to that sentence in Turkish:f{i[1].text}"}]
    )

    content = completion.choices[0].message['content']
    content = content.split("\n")
    for j in content:
        similar_sentence = j[j.find(".")+2:]
        similar_sentence = pd.DataFrame([[similar_sentence, i[1].label]], columns=['text','label'])
        df_similar = pd.concat([df_similar, similar_sentence], axis=0, ignore_index=True)

df_similar.to_csv("generated_dataset.csv", index=False)
# %%
