# M362K-RecommendationEngine

This is a repository for a Movie Recommendation System using a parsed version of MovieLens **_'ml-latest'_** dataset, which contains 27,000,000 ratings and 1,100,000 tag applications applied to 58,000 movies by 280,000 users. The system uses a Singular Value Decomposition algorithm to create a model of the user and movie data in order to predict recommendations.

The movie recommendations, updated with the latest users, can be found in the **_recommendations.csv_**. Feel free to modify **_parse.py_** to parse your own custom datasets based on the 27M **_'ml-latest'_** file. The file **_test.py_** is where the dataset is loaded and the algorithm run.

