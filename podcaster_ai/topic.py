import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

def find_topics(
        podcast_descs,
        strategy='lda',
        num_top_words=10,
        max_df=0.8,
        min_df=2,
        stop_words='english',
        n_components=5,
        verbose=0,
        ):
    if strategy == 'lda':
        return _find_topics(
                podcast_descs,
                num_top_words,
                max_df,
                min_df,
                stop_words,
                n_components,
                CountVectorizer,
                LatentDirichletAllocation,
                verbose,
                )
    elif strategy == 'nmf':
        return _find_topics(
                podcast_descs,
                num_top_words,
                max_df,
                min_df,
                stop_words,
                n_components,
                TfidfVectorizer,
                NMF,
                verbose,
                )
    else:
        raise ValueError(f'Unknown strategy: {strategy}.')

def _find_topics(
    podcast_descs,
    num_top_words,
    max_df,
    min_df,
    stop_words,
    n_components, 
    vectorizer,
    model_class,
    verbose,
    random_state=42
    ):
    # First vectorize descriptions
    vectorizer = vectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words)
    # Create doc term matrix
    doc_term_matrix = vectorizer.fit_transform(podcast_descs)
    # Initialize and fit Model
    model = model_class(n_components=n_components, random_state=random_state, verbose=verbose)
    model.fit(doc_term_matrix)
    # Get topic probabilities
    topic_probs = model.transform(doc_term_matrix)
    topic_assignments = topic_probs.argmax(axis=1)
    topic_words = {}
    for i, topic in enumerate(model.components_):
        topic_words[i] = [vectorizer.get_feature_names()[i] for i in topic.argsort()[-num_top_words:]]
    return topic_assignments, topic_words, topic_probs
