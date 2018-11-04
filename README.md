# M362K-RecommendationEngine

This is a repository for a Movie Recommendation System using a parsed version of MovieLens *'ml-latest'* dataset, which contains 27,000,000 ratings and 1,100,000 tag applications applied to 58,000 movies by 280,000 users. The system uses a Singular Value Decomposition algorithm to create a model of the user and movie data in order to predict recommendations. The most successful model generated had a Root Mean Squared Error (RMSE) of 0.5371 and a Mean Average Error (MAE) of 0.4066.

The movie recommendations, updated with the latest users, can be found in the *recommendations.csv*. Feel free to modify *parse.py* to parse your own custom datasets based on the 27M *'ml-latest'* file. The file *test.py* is where the dataset is loaded and the algorithm run, if you are going to use it, **make sure to change** the *movies.csv* file location to accomodate to your file system.  

