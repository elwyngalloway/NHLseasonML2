# NHLseasonML2
Evolution of NHLseasonML...

### Some thoughts and ideas about where this project might explore to build on the "success" of ML1:
* #### Re-write the code to be more "harnessable" - something that can easily built into a script and run multiple times with different hyper-/parameters.
** The database queries are still hard-coded
** After the data is pulled, the remaining steps should be harnessable
* #### Use some more/different data - include position, age and draft position information, for example.
* #### Include player ID as categorical data? I don't think this makes sense!
* #### Determine if using 3 years of lag is best.
* #### Does missing data affect the results?
* #### Does one need to use different models for different player archetypes? Can one model predict a grinder's and a sniper's performance?
* #### Can more than one responding variable be predicted at the same time? - Predict multiple (all?) stats at once.
* #### How can the design of the neural network be improved?
* #### The final predictions should be probabilistic.
* #### When it comes to predicting total points, is it better to predict goals and assists separately, then combine them? Or can a points prediction do a better job?
