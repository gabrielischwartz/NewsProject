import requests
import pandas as pd
import spacy
from collections import Counter
import re
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime, timedelta

url = "https://newsapi.org/v2/everything"
params = {
    'q': 'olympics',
    'from': datetime.now() - timedelta(hours=24), 
    'sortBy': 'popularity',
    'apiKey': 'da7394bf6c484381bbf90c62c919273e',
    'source.name' : 'ESPN'
}
response = requests.get(url, params=params)
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()

    # Convert the JSON response to a DataFrame
    df = pd.json_normalize(data['articles'])
    
    df = df[['author', 'publishedAt', 'source.name', 'title', 'url', 'content']]
    
    # Rename the columns to desired names
    df.rename(columns={'publishedAt': 'date', 'source.name': 'news source', 'url': 'link'}, inplace=True)
else:
    print(f"Failed to retrieve data: {response.status_code}")
    

nlp = spacy.load('en_core_web_sm')
word_counter = Counter()

def process_text(text):
    text = re.sub(r'\r\n|\r|\n', ' ', text).lower()
    doc = nlp(text.lower())  # Convert text to lowercase and process with spaCy
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]  # get rid of stopwords
    # TODO set names as one word
    return tokens

def text_to_string(contents):
    result = ""
    for content in contents.dropna().head():
        result += content
    return result

content_string = text_to_string(df['content'])

for content in df['content'].dropna():
    tokens = process_text(content)
    word_counter.update(tokens)

if content_string:
    print(word_counter.most_common(10))

    model_id = "/Users/gabeschwartz/Documents/GitHub/NewsProject/Meta-Llama-3.1-8B-Instruct" 
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Define the input text
    input_text = text_to_string(content_string + "Create 1 summary of all of these articles")

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate text with streaming
    generated_text = ""
    for output in model.generate(input_ids, max_new_tokens=256, do_sample=True):
        generated_text += tokenizer.decode(output, skip_special_tokens=True)
        print(generated_text)

else:
    print("There is no content")

    #print(outputs[0]["generated_text"][-1]['content'])


