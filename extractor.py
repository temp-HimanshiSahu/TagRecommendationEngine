import requests
from bs4 import BeautifulSoup as bs
import re
from collections import defaultdict, Counter
from fuzzywuzzy import fuzz
from rich.console import Console
from rich.table import Table
import yake
import html
import pandas as pd
import spacy
import textwrap
from concurrent.futures import ThreadPoolExecutor
import os
from gensim.models import KeyedVectors

# Initialize rich console
console = Console()

# Initialize YAKE with a higher number of keywords
language = "en"
max_ngram_size = 3  # Consider up to 3-grams
deduplication_threshold = 0.9
numOfKeywords = 50  # Increase this number to fetch more keywords
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords)

# Define base directory path
base_dir = r'C:\Users\Greynium\Tag Reccomandation System'

# Load tags from CSV
csv_file_path = os.path.join(base_dir, 'updated_categorywise_tags.csv')
df = pd.read_csv(csv_file_path)
csv_tags = df['Tag'].str.lower().tolist()

# Load tags from Excel file for matching
excel_file_path = os.path.join(base_dir, 'All Tags.xlsx')
df_excel = pd.read_excel(excel_file_path)
excel_tags = df_excel['Tag'].dropna().astype(str).str.lower().tolist()  # Ensure non-null strings
article_counts = df_excel.set_index('Tag')['#Articles'].to_dict()

# Load the FastText model
model_path = r'C:\Users\Greynium\Tag Reccomandation System\wiki-news-300d-1M.vec'
word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=False)

# Create inverted index
inverted_index = defaultdict(list)
for tag in csv_tags:
    for word in tag.split():
        inverted_index[word].append(tag)

# Load SpaCy model for POS tagging and NER
nlp = spacy.load("en_core_web_lg")

# Function to fetch XML content
def fetch_xml_content(url):
    source = requests.get(url)
    return source.text

# Function to clean HTML tags and entities from content
def clean_content(content):
    cleaned_content = re.sub(r"<.*?>|(&[^;]+;)", " ", html.unescape(content))
    cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
    return cleaned_content.strip()

# Function to extract words from the weblink
def extract_words_from_weblink(weblink):
    weblink = weblink.split('/')[-1]  # Get the last part of the URL
    weblink = weblink.replace('-', ' ')  # Replace dashes with spaces
    weblink = re.sub(r'\d+', '', weblink)  # Remove numbers
    weblink = re.sub(r'\.html', '', weblink)  # Remove the '.html' extension
    return weblink.strip().lower()

# Function to extract entities using SpaCy
def extract_entities(text):
    doc = nlp(text)
    entities = {ent.text.lower(): ent.label_ for ent in doc.ents if ent.label_ in ["GPE", "LOC", "PERSON"]}
    return entities

# Extract keywords using YAKE
def extract_yake_keywords(text, extractor):
    keywords = extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

# Function to extract years and prefixes
def extract_years_with_prefix(text):
    years = re.findall(r'\b(?:\d{4}|(?:[a-zA-Z]+\s)?\d{4})\b', text)
    return [year for year in years if re.search(r'\d{4}', year)]

# Calculate similarity between two strings
def calculate_similarity(tag1, tag2):
    return fuzz.ratio(tag1, tag2)

# Get candidate tags from the inverted index
def get_candidate_tags(suggested_tag):
    words = suggested_tag.split()
    candidate_tags = set()
    for word in words:
        if word in inverted_index:
            candidate_tags.update(inverted_index[word])
    return list(candidate_tags)

# Function to generate tags based on specified criteria
def generate_tags(title, keywords_with_scores, entities, existing_tags):
    title_words = set(title.split())
    generated_tags = set(entities.keys())

    for kw in keywords_with_scores:
        words = kw.split()
        if any(word in title_words for word in words):
            generated_tags.add(kw)

    # Clean generated tags
    cleaned_tags = [tag for tag in generated_tags if tag not in existing_tags]

    return list(cleaned_tags)

# Function to filter keywords based on title similarity
def filter_keywords_by_title(keywords, title_words, threshold=0.5):
    filtered_keywords = []
    for keyword in keywords:
        for title_word in title_words:
            if calculate_similarity(keyword, title_word) > threshold:
                filtered_keywords.append(keyword)
                break
    return filtered_keywords

# Function to generate suggested tags based on title words
def generate_suggested_tags(yake_keywords, ner_keywords, title_words):
    suggested_tags = set()

    for keywords in [yake_keywords, ner_keywords]:
        for keyword in keywords:
            for title_word in title_words:
                if calculate_similarity(keyword, title_word) > 0.5 or re.search(r'\b\d{4}\b', keyword):
                    suggested_tags.add(keyword)
                    break

    return list(suggested_tags)

# Function to merge similar tags and select the longest one
def merge_similar_tags(tags, threshold=0.8):
    unique_tags = set(tags)
    merged_tags = set()
    
    while unique_tags:
        tag = unique_tags.pop()
        cluster = [tag]
        
        for other_tag in list(unique_tags):
            if calculate_similarity(tag, other_tag) > threshold:
                cluster.append(other_tag)
                unique_tags.remove(other_tag)
        
        longest_tag = max(cluster, key=len)
        merged_tags.add(longest_tag)
    
    return list(merged_tags)

# Function to extract n-grams from text
def extract_ngrams(text, n):
    words = text.split()
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    return ngrams

# Function to get POS tags using SpaCy with parallel processing
def get_pos_tags_for_phrases(phrases):
    pos_tags = {}
    with ThreadPoolExecutor() as executor:
        results = executor.map(nlp, phrases)
        for phrase, doc in zip(phrases, results):
            pos_tag = " ".join([token.pos_ for token in doc])
            pos_tags[phrase] = pos_tag
    return pos_tags

# Function to filter n-grams based on POS tags
def filter_ngrams(ngrams, pos_tags):
    filtered_ngrams = {}
    for ngram, pos_tag in pos_tags.items():
        pos_tags_list = pos_tag.split()
        # Remove n-grams that start or end with a verb (VB)
        if pos_tags_list[0] not in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"] and pos_tags_list[-1] not in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
            filtered_ngrams[ngram] = pos_tag
    return filtered_ngrams

# Function to apply category-specific rules
def apply_category_specific_rules(category, tag_frequency, filtered_ngrams):
    if category == "horoscope":
        tag_frequency = {tag: freq for tag, freq in tag_frequency.items() if freq >= 5}
    
    elif category in ["press release", "partner content"]:
        tag_frequency = {tag: freq for tag, freq in tag_frequency.items() if freq > 6}
        tag_frequency["promotions"] = tag_frequency.get("promotions", 0) + 1
    
    elif category == "india":
        if any("lottery" in tag for tag in tag_frequency.keys()):
            tag_frequency = {tag: freq for tag, freq in tag_frequency.items() if "lottery" in tag}
            filtered_ngrams = {ngram: pos for ngram, pos in filtered_ngrams.items() if "lottery" in ngram}
    
    return tag_frequency, filtered_ngrams

# Function to detect synonyms and similar tags
def detect_synonyms_and_similar_tags(tags, threshold=0.7):
    synonym_clusters = defaultdict(list)
    for tag in tags:
        added = False
        for cluster_tag in synonym_clusters.keys():
            if cluster_tag in word_vectors and tag in word_vectors:
                similarity = word_vectors.similarity(cluster_tag, tag)
                if similarity > threshold:
                    synonym_clusters[cluster_tag].append(tag)
                    added = True
                    break
        if not added:
            synonym_clusters[tag].append(tag)
    return synonym_clusters

# Initial processing function to print extracted keywords and determine main topic
def process_xml_data_with_tables(url):
    xml_content = fetch_xml_content(url)
    if xml_content is None:
        console.print("Failed to fetch XML content.")
        return

    root = bs(xml_content, 'xml')
    items = root.find_all('Item')

    if not items:
        console.print("No items found in the XML content.")
        return

    results = []

    for item in items:
        title = item.find('Title').text.strip().lower()
        summary = item.find('Summary').text.strip().lower()
        weblink = item.find('WebLink').text.strip().lower()
        link = item.find('Link').text.strip().lower()
        tags_element = item.find('Tags')
        existing_tags = [tag.strip().lower() for tag in tags_element.text.split(',')] if tags_element else []
        category_element = item.find('Category')
        category = category_element.text.strip().lower() if category_element else "general"

        inner_xml_content = fetch_xml_content(link)
        if inner_xml_content is None:
            console.print(f"Failed to fetch inner XML content for link: {link}")
            continue

        inner_root = bs(inner_xml_content, 'xml')
        content_tag = inner_root.find('Content')
        content = clean_content(content_tag.text.strip().lower()) if content_tag else ""

        combined_text = title + " " + content + " " + summary + " " + extract_words_from_weblink(weblink)

        # Extract title words
        title_words_1 = title.split()

        # First method
        yake_keywords_1 = extract_yake_keywords(combined_text, custom_kw_extractor)
        ner_keywords_1 = extract_entities(combined_text)
        filtered_yake_keywords_1 = filter_keywords_by_title(yake_keywords_1, title_words_1)
        filtered_ner_keywords_1 = filter_keywords_by_title(list(ner_keywords_1.keys()), title_words_1)
        suggested_tags_1 = generate_suggested_tags(filtered_yake_keywords_1, filtered_ner_keywords_1, title_words_1)
        merged_suggested_tags_1 = merge_similar_tags(suggested_tags_1)
        final_suggested_tags_1 = [tag for tag in merged_suggested_tags_1 if tag in csv_tags]

        # Second method
        yake_keywords_2 = extract_yake_keywords(combined_text, custom_kw_extractor)
        entities_2 = extract_entities(combined_text)
        generated_tags_2 = generate_tags(title, yake_keywords_2, entities_2, existing_tags)
        suggested_tags_2 = [tag for tag in generated_tags_2 if any(fuzz.partial_ratio(tag, csv_tag) > 80 for csv_tag in csv_tags)]

        # Combine all tags
        combined_tags = (
            yake_keywords_1 +
            list(ner_keywords_1.keys()) +
            title_words_1 +
            filtered_yake_keywords_1 +
            filtered_ner_keywords_1 +
            suggested_tags_1 +
            merged_suggested_tags_1 +
            final_suggested_tags_1 +
            yake_keywords_2 +
            list(entities_2.keys()) +
            generated_tags_2 +
            suggested_tags_2
        )

        # Extract n-grams
        bigrams = extract_ngrams(" ".join(combined_tags), 2)
        trigrams = extract_ngrams(" ".join(combined_tags), 3)
        fourgrams = extract_ngrams(" ".join(combined_tags), 4)

        ngrams = bigrams + trigrams + fourgrams
        pos_tags = get_pos_tags_for_phrases(ngrams)

        # Filter n-grams based on POS tags
        filtered_ngrams = filter_ngrams(ngrams, pos_tags)

        # Count frequency of each tag
        tag_frequency = Counter(combined_tags)
        
        # Apply category-specific rules
        tag_frequency, filtered_ngrams = apply_category_specific_rules(category, tag_frequency, filtered_ngrams)

        # Remove tags with frequency 1
        tag_frequency = {tag: freq for tag, freq in tag_frequency.items() if freq > 1}

        # Detect synonyms and similar tags
        synonym_clusters = detect_synonyms_and_similar_tags(tag_frequency.keys())

        results.append({
            "title": textwrap.fill(title, width=70),
            "existing_tags": existing_tags,
            "tag_frequency": tag_frequency,
            "filtered_ngrams": filtered_ngrams,
            "title_words": title_words_1,  # Add title words to results
            "synonym_clusters": synonym_clusters
        })

    # Print results
    for result in results:
        console.print(f"\nTitle: {result['title']}")

        # Print existing tags
        console.print(f"Existing Tags: {', '.join(result['existing_tags'])}")

        console.print("\nTag Frequency")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Tag")
        table.add_column("Frequency")
        table.add_column("Article Count")

        for tag, freq in sorted(result["tag_frequency"].items(), key=lambda item: item[1], reverse=True):
            if tag in excel_tags:
                article_count = article_counts.get(tag, "N/A")
                table.add_row(tag, str(freq), str(article_count))

        console.print(table)

        console.print("\nSynonym Clusters")
        synonym_table = Table(show_header=True, header_style="bold magenta")
        synonym_table.add_column("Cluster")
        synonym_table.add_column("Tags")

        for cluster_tag, tags in result["synonym_clusters"].items():
            tags_with_check = [
                f"{tag} {'✔️' if tag in excel_tags else '❌'}" for tag in tags
            ]
            synonym_table.add_row(cluster_tag, ", ".join(tags_with_check))

        console.print(synonym_table)
        console.print("=" * 80)

# Example usage
if __name__ == "__main__":
    url = "https://rss1.oneindia.com/xml4apps/www.oneindia.com/latest.xml"
    process_xml_data_with_tables(url)
