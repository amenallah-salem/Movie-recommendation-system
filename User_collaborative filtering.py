# -*- coding: utf-8 -*-


from surprise import KNNBasic, Reader, Dataset
import heapq
from operator import itemgetter
import csv, sys, os
from collections import defaultdict


class Films:
    film_Label = {}
    label_Film = {}
    link1 = '../../ml-latest-small/ratings_after_wrangling.csv'
    link2 = '../../ml-latest-small/movies.csv'

    def download(self):

        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))

        ratingsDataset = 0
        self.film_Label = {}
        self.label_Film = {}

        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)

        ratingsDataset = Dataset.load_from_file(self.link1, reader=reader)

        with open(self.link2, newline='', encoding='ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)  # Skip header line
            for row in movieReader:
                movieID = int(row[0])
                movieName = row[1]
                self.film_Label[movieID] = movieName
                self.label_Film[movieName] = movieID

        return ratingsDataset

    def filmLabel(self, movieID):
        if movieID in self.film_Label:
            return self.film_Label[movieID]
        else:
            return ""




"""
#######################################   DashBoard    ###########################################
"""





USER = '70'
k = 10

# Load our data set and compute the user similarity matrix
ml = Films()
data = ml.download()

trainSet = data.build_full_trainset()

sim_options = {'name': 'cosine',
               'user_based': True
               }

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()

# Get top N similar users to our test subject
# (Alternate approach would be to select users up to some similarity threshold - try it!)
testUserInnerID = trainSet.to_inner_uid(USER)
similarityRow = simsMatrix[testUserInnerID]

similarUsers = []
for innerID, score in enumerate(similarityRow):
    if (innerID != testUserInnerID):
        similarUsers.append((innerID, score))

kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])



# Get the stuff they rated, and add up ratings for each item, weighted by user similarity
candidates = defaultdict(float)
for similarUser in kNeighbors:
    innerID = similarUser[0]
    userSimilarityScore = similarUser[1]
    theirRatings = trainSet.ur[innerID]
    for rating in theirRatings:
        candidates[rating[0]] += (rating[1] / 5.0) * userSimilarityScore

# Build a dictionary of stuff the user has already seen
watched = {}
for itemID, rating in trainSet.ur[testUserInnerID]:
    watched[itemID] = 1

# Get top-rated items from similar users:
pos = 0
for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if not itemID in watched:
        movieID = trainSet.to_raw_iid(itemID)
        print(ml.filmLabel(int(movieID)), ratingSum)
        pos += 1
        if (pos > 5):
            break



