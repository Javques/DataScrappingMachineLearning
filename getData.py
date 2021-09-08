#-------------------------------------------------------
#Assignment 2
#Written by Alexis Bolduc 40126092
#For COMP 472 Section AK-X –Summer 2021
#--------------------------------------------------------
#this program will gather the information on the episodes of a series
#in order for the main program to use the episodes review links and train the data
#set

import pandas as pd
from requests import get
from bs4 import BeautifulSoup

#some codes of tv series
mrRobot = "tt4158110"
gameOfThrones = "tt0944947"
pll = "tt1578873"

#gets the episode list of each season
def getEpisodeList(seriesCode, season):
    return f"https://www.imdb.com/title/{seriesCode}/episodes?season={season}"
#get url for episode reviews
def getEpisodeLink(episodeLink):
    return f"https://www.imdb.com{episodeLink}reviews"
#creates data.csv
def fillDataCSV(seriesCode,arrayOfSeason):
    dates = []
    reviewLink =[]
    episodeName = []
    season =[]
    for i in arrayOfSeason:
        episodesURL = getEpisodeList(seriesCode,i)
        response = get(episodesURL)
        html_soup = BeautifulSoup(response.text,'html.parser')
        listEpisodes = html_soup.find_all('div', class_='image')
        airdates = html_soup.find_all('div', class_='airdate')

        for airdate in airdates:
            theDate = airdate.text
            indexOfDot = theDate.find('.')
            goodDate = theDate[indexOfDot+2:indexOfDot+6]
            dates.append(goodDate)
        for episode in listEpisodes:
            url = episode.a['href']
            reviewLink.append(getEpisodeLink(url))
            title = episode.a['title']
            episodeName.append(title)
            season.append(i)
    #creating dataframe based on result
    data_df = pd.DataFrame({'Name':episodeName,'Season':season,'Review Link':reviewLink,
    'Year':dates})
    #saving df to data.csv so it can be accessed in main program
    data_df.to_csv('data.csv')
#change parameters based on tv series code  
theSeasons = [5,6,7,8]      
fillDataCSV(gameOfThrones,theSeasons)

print("data.csv successfully generated")
#end of program thank you!