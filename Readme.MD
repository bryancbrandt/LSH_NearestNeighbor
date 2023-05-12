Final Project for Advanced algorithms in the real world

To run a specific algorithm, just uncomment the sections corresponding to
the algorithm in the Main.py file.

Change the CSV_INDEX value to correspond with a file name
If you are loading pickle files, the PICKLE_IDX must be set
for the correct filename value

Most algorithms work by first initializing the object with the file name
then preprocess the file with the option to save the data structure as a 
pickle file.
You can then load the pickle file (which is signficantly faster than preprocessing)
and run your queries using the query method.

Query values were randomly generated, but you can select which point you want to query

The BallTree, KDTree, and BruteForce use similar preprocessing methods, but return values
are different.