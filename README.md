# codesoft
# internship tasks 4
# recomendation system
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample data for movies
movies_data = {
    'Title': ['Movie 1', 'Movie 2', 'Movie 3', 'Movie 4', 'Movie 5'],
    'Genre': ['Action', 'Drama', 'Action', 'Comedy', 'Drama'],
    'Description': ['Action-packed movie with thrilling scenes',
                    'A heartfelt drama about human relationships',
                    'Exciting action movie with a twist',
                    'Hilarious comedy that will make you laugh',
                    'Compelling drama with strong character development']
}

movies_df = pd.DataFrame(movies_data)

# Function to recommend movies based on user preferences
def recommend_movies(user_preferences, movies_df):
    # Combine movie features into a single column
    movies_df['Features'] = movies_df['Genre'] + ' ' + movies_df['Description']

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(movies_df['Features'])

    # Calculate the cosine similarity between movies
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Create a DataFrame with movie titles and corresponding indices
    indices = pd.Series(movies_df.index, index=movies_df['Title']).drop_duplicates()

    # Convert user preferences into a movie feature vector
    user_preferences_vector = vectorizer.transform([user_preferences])

    # Calculate the cosine similarity between user preferences and movies
    similarity_scores = list(enumerate(cosine_sim[user_preferences_vector.nonzero()[0][0]]))

    # Sort movies based on similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top recommended movies
    top_indices = [score[0] for score in similarity_scores]

    # Return the top recommended movies
    return movies_df['Title'].iloc[top_indices]

# Example of how to use the recommendation system
user_preferences = 'Action-packed movie with a twist'
recommended_movies = recommend_movies(user_preferences, movies_df)

print("User Preferences:", user_preferences)
print("Recommended Movies:")
for movie in recommended_movies:
    print("-", movie)
