import numpy as np
import pandas as pd
import neattext.functions as nfx

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 1. LOAD DATA
# =========================
udemy = pd.read_csv('../datasets/courses.csv')
udemy = udemy.reset_index(drop=True)
# clean column names (safe)
udemy.columns = udemy.columns.str.strip()

# =========================
# 2. TEXT CLEANING
# =========================
udemy['clean_title'] = udemy['title'].apply(nfx.remove_stopwords)
udemy['clean_title'] = udemy['clean_title'].apply(nfx.remove_special_characters)

# =========================
# 3. VECTORIZATION
# =========================
cv = CountVectorizer()
vector = cv.fit_transform(udemy['clean_title'])

# =========================
# 4. INDEX MAPPING
# =========================
course_index = pd.Series(
    udemy.index,
    index=udemy['title']
).drop_duplicates()

# =========================
# 5. RECOMMENDATION FUNCTION
# =========================
def recommend(course_title, n=5):
    if course_title not in course_index:
        return "Course not found in dataset."

    idx = course_index[course_title]

    # cosine similarity for this one item
    sim_scores = cosine_similarity(vector[idx], vector).flatten()

    # safe top indices
    top_indices = np.argsort(sim_scores)[::-1]

    # remove itself
    top_indices = top_indices[top_indices != idx]

    # take only n
    top_indices = top_indices[:n]

    recs = udemy.iloc[top_indices].copy()
    recs['similarity'] = sim_scores[top_indices]

    return recs[['title', 'similarity']]

# =========================
# 6. TEST IT
# =========================
print(recommend("Python for Data Science and Machine Learning Bootcamp", 5))