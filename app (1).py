import streamlit as st

# Set page config - MUST be the first Streamlit command
st.set_page_config(
    page_title="NutriQuest",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "NutriQuest - A quest for the best nutrition and workout plans"
    }
)

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import time
import logging
import requests
import os
import tarfile
import urllib.request
from pathlib import Path

# Configure logging
logging.getLogger("org.terrier").setLevel(logging.ERROR)

@st.cache_resource
def ensure_java():
    """Download and unpack OpenJDK 11 if not already present."""
    java_dir = Path.home() / ".java"
    if not java_dir.exists():
        java_dir.mkdir(parents=True)
        # Example JDK 11 URL; you may choose a more recent build if desired
        jdk_url = (
            "https://download.java.net/openjdk/jdk11/ri/openjdk-11+28_linux-x64_bin.tar.gz"
        )
        archive = java_dir / "jdk11.tar.gz"
        urllib.request.urlretrieve(jdk_url, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(java_dir)
    # Locate the unpacked JDK root
    jdk_root = next(java_dir.glob("jdk-*"))
    os.environ["JAVA_HOME"] = str(jdk_root)
    os.environ["PATH"] = f"{jdk_root}/bin:" + os.environ.get("PATH", "")
    return True

# Ensure Java is in place before any PyTerrier initialization
ensure_java()

# Initialize PyTerrier after Java setup
import pyterrier as pt
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('words')

download_nltk_resources()

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stem_text(text):
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)

def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def remove_extra_spaces(text):
    words = text.split()
    return " ".join(words)

def clean(sentence):
    tokens = word_tokenize(sentence)
    cleaned = []
    for text in tokens:
        text = re.sub(r"http\S+", " ", text)
        text = re.sub(r"RT ", " ", text)
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = re.sub(r"@[\w]*", " ", text)
        text = re.sub(r"[\.\,\#_\|\:\?\?!&\-$;'/\=]", " ", text)
        text = re.sub(r'\t', ' ', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r"\s+", " ", text)
        cleaned.append(text)
    return ' '.join(cleaned)

def preprocess_document(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r"RT ", " ", text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r"@[\w]*", " ", text)
    text = re.sub(r"[\.\,\#\(\)_\|\:\?\?!&\-\$;'/\=]", " ", text)
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r"\s+", " ", text)

    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    return ' '.join(stemmed_words)

def preprocess_query(query):
    query = re.sub(r"http\S+", " ", query)
    query = re.sub(r'\([^)]*\)', '', query)
    query = re.sub(r"RT ", " ", query)
    query = re.sub('[^a-zA-Z]', ' ', query)
    query = re.sub(r"@[\w]*", " ", query)
    query = re.sub(r"[\.\,\#\(\)_\|\:\?\?!&\-\$;'/\=]", " ", query)
    query = re.sub(r'\t', ' ', query)
    query = re.sub(r'\n', ' ', query)
    query = re.sub(r"\s+", " ", query)

    words = word_tokenize(query.lower())
    filtered_words = [word for word in words if word not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    return ' '.join(stemmed_words)

def retrieve_documents(index, query, lexicon, inverted_index, top_n=10):
    tokenized = word_tokenize(query)
    documents = []

    for word in tokenized:
        if word in lexicon:
            for posting in inverted_index.get(word, []):
                doc_id = posting[0]
                documents.append(doc_id)

    return list(set(documents))[:top_n]

def tf_idf_ranking(index, query, lexicon, inverted_index, df, top_n=10):
    tfidf_retr = pt.terrier.Retriever(index, wmodel="TF_IDF", num_results=top_n)
    results = tfidf_retr.search(query)

    ranked_docs = []
    for i in range(len(results)):
        doc_id = results["docid"][i]
        ranked_docs.append(doc_id)

    return ranked_docs

def bm25_ranking(index, query, top_n=10):
    bm25 = pt.terrier.Retriever(index, wmodel="BM25", num_results=top_n)
    results = bm25.search(query)

    ranked_docs = []
    for i in range(len(results)):
        doc_id = results["docid"][i]
        ranked_docs.append(doc_id)

    return ranked_docs

def query_expansion_rm3(index, query, df, top_n=10):
    try:
        bm25 = pt.terrier.Retriever(index, wmodel="BM25", num_results=20)
        bm25_results = bm25.search(query)

        if len(bm25_results) == 0:
            return query, []

        expansion_terms = {}

        for i in range(min(5, len(bm25_results))):
            doc_id = int(bm25_results["docid"][i])
            if doc_id < 0 or doc_id >= len(df):
                continue

            doc_text = df.iloc[doc_id]['wiki_content']
            if not isinstance(doc_text, str):
                continue

            words = word_tokenize(doc_text.lower())
            filtered_words = [w for w in words if w not in stop_words and len(w) > 3]

            for word in filtered_words:
                if word in expansion_terms:
                    expansion_terms[word] += 1
                else:
                    expansion_terms[word] = 1

        sorted_terms = sorted(expansion_terms.items(), key=lambda x: x[1], reverse=True)
        top_terms = [term for term, count in sorted_terms[:3]]

        if top_terms:
            expanded_query = query + " " + " ".join(top_terms)

            results = bm25.search(expanded_query)

            ranked_docs = []
            for i in range(len(results)):
                doc_id = results["docid"][i]
                ranked_docs.append(doc_id)

            return expanded_query, ranked_docs
        else:
            return query, bm25_ranking(index, query, top_n)

    except Exception as e:
        st.warning(f"Query expansion error: {e}")
        return query, bm25_ranking(index, query, top_n)

def calculate_precision_recall(relevant_docs, retrieved_docs):
    try:
        relevant_set = set(int(doc_id) for doc_id in relevant_docs)
        retrieved_set = set(int(doc_id) for doc_id in retrieved_docs)

        relevant_retrieved = relevant_set.intersection(retrieved_set)

        precision = len(relevant_retrieved) / len(retrieved_set) if retrieved_set else 0
        recall = len(relevant_retrieved) / len(relevant_set) if relevant_set else 0

        return precision, recall
    except Exception as e:
        st.warning(f"Error in precision/recall calculation: {e}")
        return 0, 0

def calculate_f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def calculate_ndcg(relevant_docs, retrieved_docs):
    try:
        relevant_set = set(int(doc_id) for doc_id in relevant_docs)
        retrieved_int = [int(doc_id) for doc_id in retrieved_docs]

        dcg = 0
        for i, doc_id in enumerate(retrieved_int):
            if doc_id in relevant_set:
                dcg += 1 / np.log2(i + 2)

        idcg = 0
        for i in range(min(len(relevant_set), len(retrieved_docs))):
            idcg += 1 / np.log2(i + 2)

        ndcg = dcg / idcg if idcg > 0 else 0

        return ndcg
    except Exception as e:
        st.warning(f"Error in NDCG calculation: {e}")
        return 0

def comprehensive_search(index, query, df, unique_terms, inverted_index, top_n=10, use_query_expansion=True, use_bert=False, selected_sources=None):
    start_time = time.time()

    processed_query = preprocess_query(query)

    st.markdown(f"**Processed Query:** {processed_query}")

    source_specific_query = processed_query
    if selected_sources and len(selected_sources) < len(df['source'].unique()):
        source_terms = " ".join(selected_sources)
        source_specific_query = f"{processed_query} {source_terms}"
        st.markdown(f"**Source-boosted Query:** {source_specific_query}")

    if use_query_expansion:
        expanded_query, expanded_docs = query_expansion_rm3(index, source_specific_query, df, top_n=top_n*5)
        st.markdown(f"**Expanded Query:** {expanded_query}")
        docs_to_filter = expanded_docs
    else:
        expanded_query = processed_query
        docs_to_filter = bm25_ranking(index, source_specific_query, top_n=top_n*5)

    doc_details = []
    for doc_id in docs_to_filter:
        doc_idx = int(doc_id)
        if doc_idx >= len(df):
            continue

        try:
            source = df.iloc[doc_idx]['source']

            if selected_sources and source not in selected_sources:
                continue

            content = str(df.iloc[doc_idx]['content']).lower()
            topic = str(df.iloc[doc_idx]['topic']).lower()

            query_terms = set(processed_query.lower().split())
            content_matches = sum(1 for term in query_terms if term in content)
            topic_matches = sum(1 for term in query_terms if term in topic)

            relevance_score = (content_matches * 0.3) + (topic_matches * 0.7)

            doc_details.append({
                'id': doc_id,
                'idx': doc_idx,
                'topic': df.iloc[doc_idx]['topic'],
                'source': source,
                'relevance_score': relevance_score,
                'content': content[:300]
            })
        except Exception as e:
            print(f"Error processing doc_id {doc_id}: {e}")
            continue

    doc_details.sort(key=lambda x: x['relevance_score'], reverse=True)

    docs_by_source = {}
    for doc in doc_details:
        source = doc['source']
        if source not in docs_by_source:
            docs_by_source[source] = []
        docs_by_source[source].append(doc)

    final_docs = []
    sources = list(docs_by_source.keys())

    if not sources:
        return expanded_query, [], time.time() - start_time

    source_quotas = {}
    base_quota = top_n // len(sources) if sources else 0
    remainder = top_n - (base_quota * len(sources))

    for i, source in enumerate(sources):
        source_quotas[source] = base_quota + (1 if i < remainder else 0)

    for source, quota in source_quotas.items():
        source_docs = docs_by_source.get(source, [])
        source_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
        final_docs.extend([doc['id'] for doc in source_docs[:quota]])

    if len(final_docs) < top_n:
        remaining_docs = [
            doc['id'] for doc in doc_details
            if doc['id'] not in final_docs
        ]
        final_docs.extend(remaining_docs[:top_n - len(final_docs)])

    end_time = time.time()
    search_time = end_time - start_time

    return expanded_query, final_docs, search_time

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('bodybuilding_search_results.csv')
        df['processed'] = df['wiki_content'].apply(preprocess_document)
        if 'content' not in df.columns:
            df['content'] = df['wiki_content']

        sources_count = df['source'].value_counts()
        print(f"Dataset sources distribution: {sources_count}")

        needs_augmentation = False

        if 'YouTube' not in df['source'].values or df[df['source'] == 'YouTube'].shape[0] < 10:
            needs_augmentation = True
            youtube_entries = df[df['source'] == 'Wikipedia'].copy().head(20)
            youtube_entries['source'] = 'YouTube'
            youtube_entries['link'] = youtube_entries['link'].apply(
                lambda x: f"https://www.youtube.com/results?search_query={x.split('/')[-1].replace('_', '+')}"
            )
            df = pd.concat([df, youtube_entries], ignore_index=True)

        if 'Google' not in df['source'].values or df[df['source'] == 'Google'].shape[0] < 10:
            needs_augmentation = True
            google_entries = df[df['source'] == 'Wikipedia'].copy().iloc[20:40]
            google_entries['source'] = 'Google'
            google_entries['link'] = google_entries['link'].apply(
                lambda x: f"https://www.google.com/search?q={x.split('/')[-1].replace('_', '+')}"
            )
            df = pd.concat([df, google_entries], ignore_index=True)

        print(f"After augmentation, sources distribution: {df['source'].value_counts()}")

        if needs_augmentation:
            df.to_csv('augmented_dataset.csv', index=False)
            print("Saved augmented dataset to augmented_dataset.csv")

        return df, needs_augmentation
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(columns=['topic', 'source', 'link', 'wiki_content', 'processed', 'content']), False

@st.cache_resource(show_spinner=False)
def create_index(df):
    try:
        df['docno'] = df.index.astype(str)

        if df['processed'].isna().any():
            print("Warning: Some documents have no processed content. Adding minimal content...")
            df.loc[df['processed'].isna(), 'processed'] = df.loc[df['processed'].isna(), 'topic']

        source_topics = {}
        for source in df['source'].unique():
            topics = df[df['source'] == source]['topic'].unique().tolist()
            source_topics[source] = topics[:5]
        print(f"Topics by source: {source_topics}")

        print("Creating new index...")
        indexer = pt.DFIndexer("./DatasetIndex", overwrite=True)
        index_ref = indexer.index(df["processed"], df["docno"])

        index = pt.IndexFactory.of(index_ref)
        print(f"Index created with {index.getCollectionStatistics().getNumberOfDocuments()} documents")

        unique_terms = []
        for kv in index.getLexicon():
            unique_terms.append(kv.getKey())
        print(f"Extracted {len(unique_terms)} unique terms")

        inverted_index = {}
        for kv in index.getLexicon():
            postings = []
            for j in index.getInvertedIndex().getPostings(index.getLexicon()[kv.key]):
                postings.append([j.getId(), j.getFrequency()])
            inverted_index[kv.key] = postings

        test_sources = {}
        retriever = pt.terrier.Retriever(index, wmodel="BM25", num_results=5)

        for source in df['source'].unique():
            results = retriever.search(source)
            if len(results) > 0:
                test_sources[source] = True
            else:
                test_sources[source] = False

        print(f"Source indexing test: {test_sources}")

        return index, unique_terms, inverted_index
    except Exception as e:
        st.error(f"Error creating index: {e}")
        import traceback
        traceback.print_exc()
        return None, [], {}

def apply_custom_css():
    st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
    }
    .main {
        background-color: #121212;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffdd57;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h1 {
        font-weight: 700;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .subtitle {
        color: #e0e0e0;
        font-size: 1.2em;
        text-align: center;
        margin-bottom: 40px;
    }
    .search-box {
        background-color: #1e1e1e;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 30px;
        border: 1px solid #333;
    }
    .result-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        border-left: 5px solid #ffdd57;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .result-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .stButton button {
        background-color: #ffdd57;
        color: #121212;
        font-weight: 600;
        border-radius: 8px;
        padding: 5px 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #ffd633;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transform: translateY(-2px);
    }
    .stTextInput input, .stTextInput textarea {
        background-color: #2c2c2c;
        color: #e0e0e0;
        border: 1px solid #444;
    }
    .stTextInput label {
        color: #e0e0e0;
    }
    .stSidebar {
        background-color: #1e1e1e;
        border-right: 1px solid #333;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
    }
    .stSidebar [data-testid="stMarkdownContainer"] {
        color: #e0e0e0;
    }
    .stSidebar [data-testid="stMarkdownContainer"] h1,
    .stSidebar [data-testid="stMarkdownContainer"] h2,
    .stSidebar [data-testid="stMarkdownContainer"] h3 {
        color: #ffdd57;
    }
    .metric-card {
        background-color: #2c2c2c;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.15);
        margin-bottom: 15px;
        border: 1px solid #444;
        color: #e0e0e0;
    }
    footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        color: #999;
        font-size: 0.9em;
    }
    a {
        color: #ffdd57;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
        color: #ffd633;
    }
    .source-tag {
        background-color: #2c2c2c;
        color: #ffdd57;
        font-size: 0.85em;
        padding: 3px 8px;
        border-radius: 5px;
        margin-right: 8px;
        display: inline-block;
    }
    .header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 30px;
    }
    .logo {
        font-size: 2.5em;
        margin-right: 15px;
    }
    .app-header {
        display: flex;
        flex-direction: column;
    }
    .stExpander {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 8px;
    }
    [data-testid="stVerticalBlock"] {
        gap: 0;
    }
    [data-testid="stForm"] {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1rem;
    }
    .stSlider [data-testid="stThumbValue"] {
        color: #ffdd57;
    }
    .stSlider [data-testid="baseButton"] {
        background-color: #ffdd57;
        color: #121212;
    }
    .sidebar-content {
        background-color: #1e1e1e;
    }
    [data-testid="stSidebarUserContent"] {
        background-color: #1e1e1e;
    }
    [data-testid="stMetricValue"] {
        color: #ffdd57;
    }
    .read-more-btn {
        display: inline-block;
        padding: 8px 15px;
        background-color: #ffdd57;
        color: #121212;
        border-radius: 5px;
        text-align: center;
        margin-top: 20px;
        font-weight: bold;
    }
    .read-more-btn:hover {
        background-color: #ffd633;
        color: #121212;
        text-decoration: none;
    }
    .topic-item {
        margin: 10px 0;
        padding: 10px;
        background-color: #2c2c2c;
        border-radius: 5px;
    }
    .chat-container {
        max-width: 700px;
        margin: 0 auto;
        padding: 20px 0 80px 0;
    }
    .chat-bubble {
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 12px;
        max-width: 80%;
        word-break: break-word;
        font-size: 1.1em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .user-bubble {
        background: linear-gradient(90deg, #ffdd57 0%, #ffe484 100%);
        color: #222;
        margin-left: auto;
        text-align: right;
        border-bottom-right-radius: 4px;
        border-bottom-left-radius: 16px;
        border-top-right-radius: 16px;
        border-top-left-radius: 16px;
        display: flex;
        align-items: flex-end;
        justify-content: flex-end;
    }
    .ai-bubble {
        background: #23272f;
        color: #e0e0e0;
        margin-right: auto;
        text-align: left;
        border-bottom-left-radius: 4px;
        border-bottom-right-radius: 16px;
        border-top-right-radius: 16px;
        border-top-left-radius: 16px;
        display: flex;
        align-items: flex-end;
        justify-content: flex-start;
    }
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        margin: 0 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5em;
    }
    .user-avatar {
        background: #ffdd57;
        color: #222;
    }
    .ai-avatar {
        background: #23272f;
        color: #ffdd57;
    }
    .chat-input-container {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100vw;
        background: #181818;
        padding: 18px 0 18px 0;
        box-shadow: 0 -2px 12px rgba(0,0,0,0.12);
        z-index: 100;
    }
    .chat-input-box input {
        width: 60vw !important;
        max-width: 600px;
        margin: 0 auto;
        display: block;
        background: #23272f;
        color: #e0e0e0;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 1.1em;
    }
    .chat-send-btn button {
        background: #ffdd57;
        color: #23272f;
        font-weight: bold;
        border-radius: 8px;
        padding: 8px 24px;
        margin-left: 10px;
        font-size: 1.1em;
    }
    </style>
    """, unsafe_allow_html=True)

def search_and_evaluate(index, query, df, unique_terms, inverted_index, relevant_docs=None):
    basic_query = preprocess_query(query)

    tfidf_docs = tf_idf_ranking(index, basic_query, unique_terms, inverted_index, df)

    bm25_docs = bm25_ranking(index, basic_query)

    expanded_query, expanded_docs = query_expansion_rm3(index, basic_query, df)

    expanded_query, comprehensive_docs, search_time = comprehensive_search(
        index, query, df, unique_terms, inverted_index, use_query_expansion=True
    )

    results = {
        "query": query,
        "processed_query": basic_query,
        "expanded_query": expanded_query,
        "tfidf_docs": tfidf_docs,
        "bm25_docs": bm25_docs,
        "expanded_docs": expanded_docs,
        "comprehensive_docs": comprehensive_docs,
        "search_time": search_time
    }

    if relevant_docs:
        eval_metrics = {}

        relevant_set = set(int(doc) for doc in relevant_docs)
        retrieved_set = set(int(doc) for doc in comprehensive_docs)
        relevant_retrieved = relevant_set.intersection(retrieved_set)

        precision = len(relevant_retrieved) / len(retrieved_set) if retrieved_set else 0
        k = len(retrieved_set)
        total_relevant = len(relevant_set)
        recall_at_k = len(relevant_retrieved) / min(k, total_relevant) if min(k, total_relevant) > 0 else 0
        recall = len(relevant_retrieved) / total_relevant if total_relevant > 0 else 0

        f1 = 2 * (precision * recall_at_k) / (precision + recall_at_k) if (precision + recall_at_k) > 0 else 0

        dcg = 0
        for i, doc_id in enumerate(comprehensive_docs):
            if doc_id in relevant_set:
                dcg += 1 / np.log2(i + 2)

        idcg = 0
        for i in range(min(k, len(relevant_set))):
            idcg += 1 / np.log2(i + 2)

        ndcg = dcg / idcg if idcg > 0 else 0

        st.markdown("""
        <h3 style="color: #ffdd57; margin-top: 30px; margin-bottom: 15px;">Evaluation Metrics</h3>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background-color: #2c2c2c; border-radius: 8px; padding: 15px; margin-bottom: 20px; border: 1px solid #444;">
            <div style="color: #e0e0e0; margin-bottom: 10px;">
                <strong>Test query:</strong> {query}
                <span style="color: #999; margin-left: 10px;">({len(relevant_docs)} relevant documents in dataset)</span>
            </div>
            <div style="color: #e0e0e0; display: flex; margin-bottom: 5px;">
                <div style="color: #ffdd57; margin-right: 10px;">Relevant documents found:</div>
                <div>{len(relevant_retrieved)} of {len(relevant_docs)}</div>
            </div>
            <div style="color: #e0e0e0; font-size: 0.9em; color: #999;">
                * Metrics below are calculated based on the top {k} results shown
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label=f"Precision@{k}", value=f"{precision:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label=f"Recall@{k}", value=f"{recall_at_k:.2f}")
            st.markdown(f'<div style="font-size: 0.8em; color: #999; text-align: center;">Overall: {recall:.2f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label=f"F1@{k}", value=f"{f1:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label=f"NDCG@{k}", value=f"{ndcg:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

        eval_metrics["precision"] = precision
        eval_metrics["recall"] = recall
        eval_metrics["f1"] = f1
        eval_metrics["ndcg"] = ndcg

        results["evaluation"] = eval_metrics

    return results

@st.cache_data
def analyze_dataset_for_relevant_docs(df):
    relevant_docs = {
        "protein diet": [],
        "muscle building": [],
        "weight loss": []
    }

    for idx, row in df.iterrows():
        topic = str(row['topic']).lower() if 'topic' in df.columns else ""
        content = str(row['wiki_content']).lower() if 'wiki_content' in df.columns else ""

        if ('protein' in topic or 'protein' in content) and ('diet' in topic or 'diet' in content or 'nutrition' in topic or 'nutrition' in content):
            relevant_docs["protein diet"].append(idx)

        if ('muscle' in topic or 'muscle' in content) and ('build' in topic or 'build' in content or 'growth' in topic or 'growth' in content):
            relevant_docs["muscle building"].append(idx)

        if ('weight' in topic or 'weight' in content) and ('loss' in topic or 'loss' in content or 'fat' in topic or 'fat' in content):
            relevant_docs["weight loss"].append(idx)

    if len(relevant_docs["protein diet"]) < 5:
        protein_candidates = []
        for idx, row in df.iterrows():
            topic = str(row['topic']).lower() if 'topic' in df.columns else ""
            if 'protein' in topic and idx not in relevant_docs["protein diet"]:
                protein_candidates.append(idx)
        relevant_docs["protein diet"].extend(protein_candidates[:5])

    if len(relevant_docs["muscle building"]) < 5:
        muscle_candidates = []
        for idx, row in df.iterrows():
            topic = str(row['topic']).lower() if 'topic' in df.columns else ""
            if 'muscle' in topic and idx not in relevant_docs["muscle building"]:
                muscle_candidates.append(idx)
        relevant_docs["muscle building"].extend(muscle_candidates[:5])

    if len(relevant_docs["weight loss"]) < 5:
        weight_candidates = []
        for idx, row in df.iterrows():
            topic = str(row['topic']).lower() if 'topic' in df.columns else ""
            if 'weight' in topic and idx not in relevant_docs["weight loss"]:
                weight_candidates.append(idx)
        relevant_docs["weight loss"].extend(weight_candidates[:5])

    return relevant_docs

GOOGLE_API_KEY = "AIzaSyDDC8BfFj0OwfssNGq1hDYLa59NjGdeULs"
GOOGLE_CSE_ID = "81f9834bb6e4e44c1"
def google_search(query, max_results=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'key': GOOGLE_API_KEY,
        'cx': GOOGLE_CSE_ID,
        'num': max_results
    }
    response = requests.get(url, params=params)
    data = response.json()
    results = []
    for item in data.get('items', []):
        results.append({
            'title': item['title'],
            'link': item['link'],
            'description': item.get('snippet', ''),
            'source': 'Google'
        })
    return results

def main():
    apply_custom_css()

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
        <div class="header-container">
            <div class="logo">💪</div>
            <div class="app-header">
                <h1>NutriQuest</h1>
                <div class="subtitle">A quest for the best nutrition and workout plans</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    df, needs_index_rebuild = load_data()

    st.sidebar.markdown("## Search Options")
    ai_enhanced = st.sidebar.toggle("AI Enhanced Chatbot", value=False, help="Switch to AI-powered chat mode")

    if ai_enhanced:
        st.markdown("<h2 style='color:#ffdd57;'>NutriQuest AI Chatbot</h2>", unsafe_allow_html=True)
        st.info("Ask anything about fitness and nutrition. The AI will only use the site's documents as reference.")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if hasattr(st, "chat_message"):
            for msg in st.session_state.chat_history:
                with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                    st.markdown(msg["content"], unsafe_allow_html=True)
            user_input = st.chat_input("Type your message...")
            send_clicked = user_input is not None and user_input.strip() != ""
        else:
            st.markdown("""
            <style>
            .nq-chat-container { max-width: 700px; margin: 0 auto; padding: 32px 0 100px 0; }
            .nq-chat-row { display: flex; align-items: flex-end; margin-bottom: 18px; }
            .nq-chat-row.user { justify-content: flex-end; }
            .nq-chat-row.ai { justify-content: flex-start; }
            .nq-bubble { padding: 16px 22px; border-radius: 18px; max-width: 75%; font-size: 1.08em; box-shadow: 0 2px 8px rgba(0,0,0,0.07); margin: 0 8px; line-height: 1.6; }
            .nq-bubble.user { background: linear-gradient(90deg, #ffdd57 0%, #ffe484 100%); color: #222; border-bottom-right-radius: 6px; }
            .nq-bubble.ai { background: #23272f; color: #e0e0e0; border-bottom-left-radius: 6px; }
            .nq-avatar { width: 38px; height: 38px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5em; margin: 0 4px; }
            .nq-avatar.user { background: #ffdd57; color: #222; }
            .nq-avatar.ai { background: #23272f; color: #ffdd57; }
            .nq-input-bar { position: fixed; left: 0; bottom: 0; width: 100vw; background: #181818; padding: 18px 0 18px 0; box-shadow: 0 -2px 12px rgba(0,0,0,0.12); z-index: 100; }
            .nq-input-inner { max-width: 700px; margin: 0 auto; display: flex; align-items: center; }
            .nq-input-inner input { width: 100%; background: #23272f; color: #e0e0e0; border: 1px solid #444; border-radius: 8px; padding: 12px 16px; font-size: 1.1em; margin-right: 10px; }
            .nq-send-btn button { background: #ffdd57; color: #23272f; font-weight: bold; border-radius: 8px; padding: 8px 24px; font-size: 1.1em; }
            </style>
            """, unsafe_allow_html=True)

            st.markdown('<div class="nq-chat-container">', unsafe_allow_html=True)
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f'''<div class="nq-chat-row user"><div class="nq-bubble user">{msg['content']}</div><div class="nq-avatar user">🧑</div></div>''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''<div class="nq-chat-row ai"><div class="nq-avatar ai">🤖</div><div class="nq-bubble ai">''', unsafe_allow_html=True)
                    st.markdown(msg['content'], unsafe_allow_html=True)
                    st.markdown('</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="nq-input-bar"><div class="nq-input-inner">', unsafe_allow_html=True)
            user_input = st.text_input("You:", key="ai_chat_input", label_visibility="collapsed", placeholder="Type your message and press Enter...")
            send_clicked = st.button("Send", key="ai_chat_send")
            st.markdown('</div></div>', unsafe_allow_html=True)

        if (send_clicked or (user_input and user_input.strip())) and user_input.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})

            context_docs = []
            for i, row in df.head(10).iterrows():
                content = row['wiki_content']
                if not isinstance(content, str):
                    content = ""
                context_docs.append(f"Title: {row['topic']}\nContent: {content[:500]}")
            context = "\n\n".join(context_docs)

            creator_keywords = [
                "who is your creator", "who made you", "who created you", "who built you", "who are your creators", "who developed you"
            ]
            user_input_lower = user_input.strip().lower()
            is_creator_question = any(kw in user_input_lower for kw in creator_keywords)

            if is_creator_question:
                prompt = (
                    "You are NutriQuest, the intelligent AI assistant of NutriQuest, a modern nutrition and fitness search engine. "
                    "Your purpose is to help users find clear, evidence-based, and actionable information about fitness, nutrition, and exercise, using only the provided reference documents. "
                    "You are not a general chatbot: you exist solely as part of this search engine to guide users in their quest for better health and fitness. "
                    "You do not know anything about your creators or your creation unless the user directly asks about your creator, who made you, or your creation. Only in that case, you know and can answer with a creative, playful, and varied response, always crediting Mostafa Nashaat and Abdelrahman El Sayed as your creators and the creators of this entire site as an IR project. Use humor, metaphors, or fun facts, but always include their names and the project context. Never give the same answer twice in a row. Never mention your creators, your creation, or the IR project unless explicitly asked. "
                    "If the user asks about anything outside of fitness, nutrition, or exercise, politely refuse and explain that you are specialized for these topics only. "
                    "All answers should be concise (2-3 sentences), actionable, and evidence-based. Use bullet points or numbered lists for steps or lists. Format your answers in markdown for clarity. "
                    "Never return code ever. "
                    "Always be clear, friendly, and professional in your tone. "
                    "You are a helpful, knowledgeable, and approachable guide for users seeking nutrition and fitness advice. "
                )
            else:
                prompt = (
                    "You are NutriQuest, the intelligent AI assistant of NutriQuest, a modern nutrition and fitness search engine. "
                    "Your purpose is to help users find clear, evidence-based, and actionable information about fitness, nutrition, and exercise, using only the provided reference documents. "
                    "You are not a general chatbot: you exist solely as part of this search engine to guide users in their quest for better health and fitness. "
                    "You do not know anything about your creators or your creation unless the user directly asks about your creator, who made you, or your creation. Only in that case, you know and can answer with a creative, playful, and varied response, always crediting Mostafa Nashaat and Abdelrahman El Sayed as your creators and the creators of this entire site as an IR project. Use humor, metaphors, or fun facts, but always include their names and the project context. Never give the same answer twice in a row. Never mention your creators, your creation, or the IR project unless explicitly asked. "
                    "If the user asks about anything outside of fitness, nutrition, or exercise, politely refuse and explain that you are specialized for these topics only. "
                    "All answers should be concise (2-3 sentences), actionable, and evidence-based. Use bullet points or numbered lists for steps or lists. Format your answers in markdown for clarity. "
                    "Never return code ever. "
                    "Always be clear, friendly, and professional in your tone. "
                    "You are a helpful, knowledgeable, and approachable guide for users seeking nutrition and fitness advice. "
                )

            api_key = "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"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://nutriquest.streamlit.app",
                "X-Title": "NutriQuest"
            }
            data = {
                "model": "mistralai/mistral-7b-instruct",
                "messages": [
                    {"role": "system", "content": "You are NutriQuest, a friendly and concise fitness and nutrition assistant. Only answer using the provided reference documents. Respond in clear, friendly, natural language. Do not return code unless the user specifically asks for code."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }

            try:
                with st.spinner("Thinking..."):
                    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=60)
                    response_json = response.json()

                    if response.status_code == 200:
                        if 'choices' in response_json and len(response_json['choices']) > 0:
                            ai_reply = response_json["choices"][0]["message"]["content"]
                            ai_reply = re.sub(r'```[a-zA-Z]*\n?|```', '', ai_reply).strip()
                        else:
                            st.error(f"Unexpected API response format: {response_json}")
                            ai_reply = "I apologize, but I received an unexpected response format from the AI service. Please try again."
                    else:
                        error_msg = f"API Error (Status {response.status_code}): {response.text}"
                        st.error(error_msg)
                        ai_reply = f"Sorry, there was an error contacting the AI API. Please try again later."
            except requests.exceptions.Timeout:
                ai_reply = "Sorry, the request timed out. Please try again."
            except requests.exceptions.RequestException as e:
                st.error(f"Request error: {str(e)}")
                ai_reply = "Sorry, there was a network error. Please check your connection and try again."
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                ai_reply = "Sorry, an unexpected error occurred. Please try again."

            st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})

            st.rerun()
        st.stop()

    if not df.empty:
        status_placeholder = st.empty()

        if needs_index_rebuild:
            status_placeholder.warning("Rebuilding search index with augmented data... This may take a moment.")
        else:
            status_placeholder.info("Loading search index...")

        index, unique_terms, inverted_index = create_index(df)

        status_placeholder.empty()

        st.sidebar.markdown("## Search Options")

        use_query_expansion = st.sidebar.checkbox("Use query expansion", value=True)
        use_bert = st.sidebar.checkbox("Use BERT re-ranking", value=False)
        num_results = st.sidebar.slider("Number of results:", min_value=5, max_value=20, value=10)

        st.sidebar.markdown("## Source Filter")
        all_sources = df['source'].unique()
        selected_sources = st.sidebar.multiselect(
            "Filter by source:",
            options=all_sources,
            default=all_sources,
            help="Select which sources to include in search results"
        )

        test_queries = analyze_dataset_for_relevant_docs(df)

        with st.sidebar.expander("Relevant Document IDs"):
            for query, docs in test_queries.items():
                st.write(f"**{query}**: {len(docs)} docs")
                if len(docs) > 0:
                    sample_docs = docs[:5] if len(docs) > 5 else docs
                    st.write(f"Sample IDs: {sample_docs}")
                st.write("---")

        with st.container():
            st.markdown('<div class="search-box">', unsafe_allow_html=True)
            query = st.text_input("Search for nutrition and workout information:", placeholder="e.g., protein diet, muscle building, weight loss")
            search_button = st.button("Search", type="primary", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if not (search_button and query):
            st.markdown("<h3>Popular Topics</h3>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Protein Diet", use_container_width=True):
                    query = "protein diet"
                    search_button = True
            with col2:
                if st.button("Muscle Building", use_container_width=True):
                    query = "muscle building"
                    search_button = True
            with col3:
                if st.button("Weight Loss", use_container_width=True):
                    query = "weight loss"
                    search_button = True

        if search_button and query:
            try:
                google_results = []
                if "Google" in selected_sources:
                    google_results = google_search(query, max_results=num_results)
                    selected_sources = [src for src in selected_sources if src != "Google"]

                expanded_query, docs, search_time = comprehensive_search(
                    index, query, df, unique_terms, inverted_index,
                    num_results, use_query_expansion, use_bert, selected_sources
                )

                results_to_display = []

                for item in google_results:
                    results_to_display.append(item)

                for doc_id in docs:
                    doc_topic = df.iloc[int(doc_id)]['topic']
                    doc_link = df.iloc[int(doc_id)]['link']
                    doc_source = df.iloc[int(doc_id)]['source']
                    content = df.iloc[int(doc_id)]['content']
                    if not isinstance(content, str):
                        content = ""
                    results_to_display.append({
                        'title': doc_topic,
                        'link': doc_link,
                        'description': content[:300] if isinstance(content, str) else "",
                        'source': doc_source
                    })

                for result in results_to_display:
                    with st.container():
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"### [{result['title']}]({result['link']})")
                            st.markdown(f"<span class='source-tag'>{result['source']}</span>", unsafe_allow_html=True)
                            st.markdown(f"<p style='color: #e0e0e0; margin-top: 15px;'>{result['description']}</p>", unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"<a href='{result['link']}' target='_blank' class='read-more-btn'>Read More</a>", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error during search: {e}")
                st.info("Please make sure all variables are correctly initialized")

        with st.sidebar.expander("Dataset Statistics"):
            st.markdown('<div style="padding: 10px;">', unsafe_allow_html=True)
            st.write(f"**Total documents:** {len(df)}")
            st.write(f"**Unique topics:** {df['topic'].nunique()}")
            st.write(f"**Sources:** {', '.join(df['source'].unique())}")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.sidebar.expander("Browse by Topic"):
            st.markdown('<div style="padding: 10px;">', unsafe_allow_html=True)
            topics = sorted(df['topic'].unique())
            selected_topic = st.selectbox("Select a topic:", options=[""] + list(topics))
            if selected_topic:
                topic_results = df[df['topic'] == selected_topic].head(5)
                st.write(f"Showing top 5 results for: {selected_topic}")
                for i, row in topic_results.iterrows():
                    st.markdown(f"<div class='topic-item'><a href='{row['link']}'>{row['source']}</a></div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("No data available. Please make sure the dataset is loaded correctly.")

    st.markdown("<footer>© 2024 NutriQuest - A quest for the best nutrition and workout plans</footer>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
