import requests
import pandas as pd
import spacy
from collections import Counter
import re
import transformers
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from datetime import datetime, timedelta, date

def fetch_sources():
    url = 'https://newsapi.org/v2/sources'
    params = {'apiKey': 'da7394bf6c484381bbf90c62c919273e'}
    response = requests.get(url, params=params)
    data = response.json()
    return {source['name']: source['id'] for source in data['sources']}

def keywords():
    word_counter = Counter()
    pass


import requests
import pandas as pd
from datetime import datetime, timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def summarize(input_text, source_ids):
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': input_text,
        'from': (datetime.now() - timedelta(hours=24)).isoformat(),
        'sortBy': 'popularity',
        'apiKey': 'da7394bf6c484381bbf90c62c919273e',
        'sources': ','.join(source_ids)
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        
        # Convert the JSON response to a DataFrame
        df = pd.json_normalize(data['articles'])
        
        # Check if the DataFrame is empty
        if df.empty:
            return "No articles found for the given query."
        
        df = df[['author', 'publishedAt', 'source.name', 'title', 'url', 'content']]
        
        # Rename the columns to desired names
        df.rename(columns={'publishedAt': 'date', 'source.name': 'news source', 'url': 'link'}, inplace=True)
        
        # Extract the content for summarization
        articles_content = " ".join(df['content'].dropna())

        # Initialize the model and tokenizer
        model_name = "facebook/bart-large-cnn"
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name)
        
        # Create input for summarization
        inputs = tokenizer.encode("Summarize this: " + articles_content, return_tensors="pt", max_length=1024, truncation=True)
        
        # Generate the summary
        summary_ids = model.generate(inputs, max_length=1024, min_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Include the top 3 articles
        top_articles = df.head(3)[['title', 'link']].to_dict(orient='records')
        articles_str = "\n".join([f"Title: {article['title']}\nLink: {article['link']}" for article in top_articles])
        
        # Combine summary and top articles
        final_summary = f"{summary}\n\nTop 3 articles used:\n{articles_str}"
        
        return final_summary
    
    else:
        return f"Failed to retrieve data: {response.status_code}"

def process_text(text, nlp):
    text = re.sub(r'\r\n|\r|\n', ' ', text).lower()
    doc = nlp(text.lower())  # Convert text to lowercase and process with spaCy
    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]  # get rid of stopwords
    # TODO set names as one word
    return tokens

def text_to_string(contents):
    result = ""
    for content in contents.dropna().head():
        result += content + "\n\n"
    return result