# One Piece Character Appearance Predictions
I thought it might be fun to see if you could predict what episode a One Piece character may appear based on when they appear in the manga. This dataset is from Kaggle and can be found [here](https://www.kaggle.com/datasets/michau96/one-piece-characters-and-chapters).
The model has a mean absolute error of 36.05 and an R-squared value of ~0.98, meaning that there is a strong correlation between the predictors and the results, but the predictive power of these features leave something to be desired. This relatively high mean absolute error is expected given the very few predictors that were fed into the model and is frankly better than I had guessed it would be. Just for fun I threw in a calculation of the percent of predictions that were within 50 episodes of the actual episode. This felt like a good amount of buffer room (given the egregious number of episodes in this anime) to see if the model was on track for being 'good enough' given how high the mean absolute error was. There was a 92.81% accuracy for that, so with the buffer room the model fairs much better. I'd like to revisit this at some point with more predictors; a predictor like what arc the chapter or episode are within may be useful. 

In terms of feature selection the data only had name, chapter, episode, year (of chapter release), and notes. The final model used chapter and year as a predictor. Including a character's name in the model didn't affect mean absolute error, but did decrease the R-squared value to ~0.94. The accuracy within 50 episodes dropped to 91.99%. Using only chapter as a predictor brought the mean absolute error up to ~41.36 and the R-squared value down to 0.93. Most notably, the accuracy within 50 episodes was brought down to just 79.67%.

TLDR: The chapter a character appears in the manga is not a good predictor of the anime episode in which they will appear. However, including the year and a buffer of 50 episodes in this model significantly improves the model performance.
