To run the whole project, you only need the remove.txt file,
remove.txt file is a file with all the stop words that are to be removed.

********Part 1.1************************
You first need to run getData.py in order to get the data.csv file in your project,
this program can take any tv series code and scrape its list of episode in order to 
fill the table for episode review links
data.csv is the table with all the episodes and their review links

The default tv series in getData.py is Game of thrones with seasons 5 to 8 because it has 
a nice ratio of good and bad reviews
you can change the series, and the number of seasons
by chaning the parameter in fillDataCSV at line 56

******************Part1.2-Part1.3**************************
After running getData.py, you can now run MLReview.py that will take the file 
created in data.csv and parse the episode reviews and create a model 

By running this program, several files will be created 

Reviews.csv is a file that contains all the reviews of all episodes with their corresponding score

model.txt is the base probabilistic model

ActuallyRemoved.txt is a file with all the words that have been removed during execution

result.txt is a file with the prediction results based on testing the trained data set with the test data set

********End of first part*********

I'm doing task 2.2

the resulting files are:

smooth-model.txt -> probabilistic model with smoothing of 1.6
smooth-result.txt -> prediction results based on testing with model of smoothing 1.6

Also, a graph will be displayed in order to show the difference of prediction correctness percentage over different
smoothing factors
 
Library used:

numpy for mathematical operations
pandas for building dataframe
matplotlib to plot the graph
requests to access html content
beautifulsoup to parse html content