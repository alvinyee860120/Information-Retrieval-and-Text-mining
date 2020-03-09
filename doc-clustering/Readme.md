**Doc clustering with HAC clustering:**
1. data source: Text collection with 1095 news documents.
2. distribute the docs into K clusters, where K = 8, 13, and 20.

**Note:**
1. Documents are represented as normalized tf-idf vectors.
2. Use cosine similarity for pair-wise document similarity.
3. Similarity measure between clusters can be: single-link, complete-link, group-average, and centroid similarity.

**My method:** Using complete-link as my kernal of HAC clustering.
