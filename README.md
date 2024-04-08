
# Enhanced Search Engine

This project presents a robust search engine leveraging advanced retrieval methods. Our approach integrates an inverted index, BM25 scoring, and PageRank algorithms to deliver relevant search results efficiently.

## Project Overview

Utilizing a Wikipedia dump as our dataset, we meticulously parse and process the data to construct an efficient inverted index. The search engine is optimized to mitigate common challenges associated with document retrieval, ensuring precision and relevance.

## Processing Pipeline

### Data Preprocessing
- **Parsing**: Converts the wiki dump into a structured index with corresponding posting lists.
- **Stopword Removal**: Cleanses the text by eliminating common English stopwords.
- **Stemming**: Applies Porter Stemmer to standardize word variants, enhancing the matching process.

### BM25 Scoring
We adopt BM25 due to its nuanced handling of document length and term frequency. This scoring mechanism adjusts for the potential overvaluation of lengthy documents and normalizes term frequency to prevent bias.

### Optimization Strategy
BM25 calculations are precomputed within each posting list entry. This foresight significantly reduces computational demands at query time, enabling rapid retrieval by simply adjusting the precalculated values based on the user's query.

### PageRank Computation
The search engine incorporates PageRank scores to provide an alternative similarity measure. While parsing, we extract anchor texts to determine the document's PageRank, encapsulating the corpus's link structure in a `{docid: rank}` dictionary.

## Query Execution

When processing a query:
1. The text undergoes stemming and stopword removal.
2. BM25 scores are computed to determine the top 200 pages.
3. A weighted combination of BM25 and PageRank scores is applied to prioritize results.

The search engine then presents the 30 most relevant documents from the refined set.

## Evaluation Metric

To measure performance, we employ a harmonized metric that blends Precision at 5 (P@5) and F1 score at 30 (F1@30). This balance ensures that both immediate and broader relevance is assessed, prioritizing systems that excel in early precision and sustained quality across results.

## Performance Trade-offs

In our tests, different indexing strategies revealed trade-offs between response time and quality:

- **Title Index Only**: Yielded a swift average response time of 0.6 seconds with a quality score of 0.29.
- **With Body Index**: The introduction of the body index increased the average time to 6.47 seconds but improved the quality score to 0.37.

Despite the higher quality, the substantial increase in retrieval time led to the decision to prioritize the title index, striking a balance between speed and relevance.
