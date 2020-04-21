# -*- coding: utf-8 -*-



from surprise import NormalPredictor, Reader, Dataset, KNNBasic, KNNBaseline, accuracy
from surprise.model_selection import train_test_split, LeaveOneOut
from collections import defaultdict
import csv, sys, os
import numpy as np
import itertools
import random




class EndorsementMeasures:

    def MAE(predictions):
        return accuracy.mae(predictions, verbose=False)

    def RMSE(predictions):
        return accuracy.rmse(predictions, verbose=False)

    def DominantRec(predictions, n=5, minimumRating=4.0):
        topN = defaultdict(list)


        for userID, movieID, actualRating, estimatedRating, _ in predictions:
            if (estimatedRating >= minimumRating):
                topN[int(userID)].append((int(movieID), estimatedRating))

        for userID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(userID)] = ratings[:n]

        return topN

    def ShotQuota(topNPredicted, leftOutPredictions):
        hits = 0
        total = 0

        # For each left-out rating
        for leftOut in leftOutPredictions:
            userID = leftOut[0]
            leftOutMovieID = leftOut[1]
            # Is it in the predicted top 10 for this user?
            hit = False
            for movieID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutMovieID) == int(movieID)):
                    hit = True
                    break
            if (hit) :
                hits += 1

            total += 1

        # Compute overall precision
        return hits/total

    def AggregateShotQuota(topNPredicted, leftOutPredictions, ratingCutoff=0):
        hits = 0
        total = 0

        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Only look at ability to recommend things the users actually liked...
            if (actualRating >= ratingCutoff):
                # Is it in the predicted top 10 for this user?
                hit = False
                for movieID, predictedRating in topNPredicted[int(userID)]:
                    if (int(leftOutMovieID) == movieID):
                        hit = True
                        break
                if (hit) :
                    hits += 1

                total += 1

        # Compute overall precision
        return hits/total

    def ValuationShotQuota(topNPredicted, leftOutPredictions):
        hits = defaultdict(float)
        total = defaultdict(float)

        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hit = False
            for movieID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutMovieID) == movieID):
                    hit = True
                    break
            if (hit) :
                hits[actualRating] += 1

            total[actualRating] += 1

        # Compute overall precision
        for rating in sorted(hits.keys()):
            print (rating, hits[rating] / total[rating])

    def MedianComplementaryShotQuota(topNPredicted, leftOutPredictions):
        summation = 0
        total = 0
        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hitRank = 0
            rank = 0
            for movieID, predictedRating in topNPredicted[int(userID)]:
                rank = rank + 1
                if (int(leftOutMovieID) == movieID):
                    hitRank = rank
                    break
            if (hitRank > 0) :
                summation += 1.0 / hitRank

            total += 1

        return summation / total

    # What percentage of users have at least one "good" recommendation
    def CustomerReport(topNPredicted, numUsers, ratingThreshold=0):
        hits = 0
        for userID in topNPredicted.keys():
            hit = False
            for movieID, predictedRating in topNPredicted[userID]:
                if (predictedRating >= ratingThreshold):
                    hit = True
                    break
            if (hit):
                hits += 1

        return hits / numUsers

    def Dissimilarity(topNPredicted, simsAlgo):
        n = 0
        total = 0
        simsMatrix = simsAlgo.compute_similarities()
        for userID in topNPredicted.keys():
            pairs = itertools.combinations(topNPredicted[userID], 2)
            for pair in pairs:
                movie1 = pair[0][0]
                movie2 = pair[1][0]
                innerID1 = simsAlgo.trainset.to_inner_iid(str(movie1))
                innerID2 = simsAlgo.trainset.to_inner_iid(str(movie2))
                similarity = simsMatrix[innerID1][innerID2]
                total += similarity
                n += 1

        S = total / n
        return (1-S)

    def Freshness(topNPredicted, rankings):
        n = 0
        total = 0
        for userID in topNPredicted.keys():
            for rating in topNPredicted[userID]:
                movieID = rating[0]
                rank = rankings[movieID]
                total += rank
                n += 1
        return total / n


class AppraiseMethod:

    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name

    def Appraise(self, evaluationData, doTopN, n=5, verbose=True):
        metrics = {}
        # Compute accuracy
        if (verbose):
            print("Evaluating accuracy...")
        self.algorithm.fit(evaluationData.Training())
        predictions = self.algorithm.test(evaluationData.Testing())
        metrics["RMSE"] = EndorsementMeasures.RMSE(predictions)
        metrics["MAE"] = EndorsementMeasures.MAE(predictions)

        if (doTopN):
            # Appraise top-10 with Leave One Out testing
            if (verbose):
                print("Evaluating top-N with leave-one-out...")
            self.algorithm.fit(evaluationData.LeaveOneOutTraing())
            leftOutPredictions = self.algorithm.test(evaluationData.LeaveOneOutTesting())
            # Build predictions for all ratings not in the training set
            allPredictions = self.algorithm.test(evaluationData.LeaveOneOutTestingOppositeTesting())
            # Compute top 10 recs for each user
            topNPredicted = EndorsementMeasures.DominantRec(allPredictions, n)
            if (verbose):
                print("Computing hit-rate and rank metrics...")
            # See how often we recommended a movie the user actually rated
            metrics["HR"] = EndorsementMeasures.ShotQuota(topNPredicted, leftOutPredictions)
            # See how often we recommended a movie the user actually liked
            metrics["cHR"] = EndorsementMeasures.AggregateShotQuota(topNPredicted, leftOutPredictions)
            # Compute ARHR
            metrics["ARHR"] = EndorsementMeasures.MedianComplementaryShotQuota(topNPredicted, leftOutPredictions)

            # Appraise properties of recommendations on full training set
            if (verbose):
                print("Computing recommendations with full data set...")
            self.algorithm.fit(evaluationData.EntireTraining())
            allPredictions = self.algorithm.test(evaluationData.EntireTesting())
            topNPredicted = EndorsementMeasures.DominantRec(allPredictions, n)
            if (verbose):
                print("Analyzing coverage, diversity, and novelty...")
            # Print user coverage with a minimum predicted rating of 4.0:
            metrics["Coverage"] = EndorsementMeasures.CustomerReport(topNPredicted,
                                                                  evaluationData.EntireTraining().n_users,
                                                                  ratingThreshold=4.0)
            # Measure diversity of recommendations:
            metrics["Dissimilarity"] = EndorsementMeasures.Dissimilarity(topNPredicted, evaluationData.Closeness())

            # Measure novelty (average popularity rank of recommendations):
            metrics["Freshness"] = EndorsementMeasures.Freshness(topNPredicted,
                                                            evaluationData.Celebrity())

        if (verbose):
            print("Analysis complete.")

        return metrics

    def Label(self):
        return self.name

    def Method(self):
        return self.algorithm


class Appraisal:

    def __init__(self, data, popularityRankings):
        self.rankings = popularityRankings

        # Build a full training set for evaluating overall properties
        self.fullTrainSet = data.build_full_trainset()
        self.fullAntiTestSet = self.fullTrainSet.build_anti_testset()

        # Build a 75/25 train/test split for measuring accuracy
        self.trainSet, self.testSet = train_test_split(data, test_size=.25, random_state=1)

        # Build a "leave one out" train/test split for evaluating top-N recommenders
        # And build an anti-test-set for building predictions
        LOOCV = LeaveOneOut(n_splits=1, random_state=1)
        for train, test in LOOCV.split(data):
            self.LOOCVTrain = train
            self.LOOCVTest = test

        self.LOOCVAntiTestSet = self.LOOCVTrain.build_anti_testset()

        # Compute similarty matrix between items so we can measure diversity
        sim_options = {'name': 'cosine', 'user_based': False}
        self.simsAlgo = KNNBaseline(sim_options=sim_options)
        self.simsAlgo.fit(self.fullTrainSet)

    def EntireTraining(self):
        return self.fullTrainSet

    def EntireTesting(self):
        return self.fullAntiTestSet

    def OppositeTesting(self, USER):
        trainset = self.fullTrainSet
        fill = trainset.global_mean
        anti_testset = []
        u = trainset.to_inner_uid(str(USER))
        user_items = set([j for (j, _) in trainset.ur[u]])
        anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                         i in trainset.all_items() if
                         i not in user_items]
        return anti_testset

    def Training(self):
        return self.trainSet

    def Testing(self):
        return self.testSet

    def LeaveOneOutTraing(self):
        return self.LOOCVTrain

    def LeaveOneOutTesting(self):
        return self.LOOCVTest

    def LeaveOneOutTestingOppositeTesting(self):
        return self.LOOCVAntiTestSet

    def Closeness(self):
        return self.simsAlgo

    def Celebrity(self):
        return self.rankings


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


    def celebrity(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.link1, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                movieID = int(row[1])
                ratings[movieID] += 1
        rank = 1
        for movieID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[movieID] = rank
            rank += 1
        return rankings


    def filmLabel(self, movieID):
        if movieID in self.film_Label:
            return self.film_Label[movieID]
        else:
            return ""


class referee:
    algorithms = []

    def __init__(self, dataset, rankings):
        ed = Appraisal(dataset, rankings)
        self.dataset = ed

    def SumMethod(self, algorithm, name):
        alg = AppraiseMethod(algorithm, name)
        self.algorithms.append(alg)

    def Appraise(self, doTopN):
        results = {}
        for algorithm in self.algorithms:
            print("Evaluating ", algorithm.Label(), "...")
            results[algorithm.Label()] = algorithm.Appraise(self.dataset, doTopN)

        # Print results
        print("\n")

        if (doTopN):
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                "Algorithm", "RMSE", "MAE", "HR", "cHR", "ARHR", "Coverage", "Dissimilarity", "Freshness"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                    name, metrics["RMSE"], metrics["MAE"], metrics["HR"], metrics["cHR"], metrics["ARHR"],
                    metrics["Coverage"], metrics["Dissimilarity"], metrics["Freshness"]))
        else:
            print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f}".format(name, metrics["RMSE"], metrics["MAE"]))

        print("\nLegend:\n")
        print("RMSE:      Root Mean Squared Error. Lower values mean better accuracy.")
        print("MAE:       Mean Absolute Error. Lower values mean better accuracy.")
        if (doTopN):
            print("HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better.")
            print(
                "cHR:       Cumulative Hit Rate; hit rate, confined to ratings above a certain threshold. Higher is better.")
            print(
                "ARHR:      Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. Higher is better.")
            print(
                "Coverage:  Ratio of users for whom recommendations above a certain threshold exist. Higher is better.")
            print(
                "Dissimilarity: 1-S, where S is the average similarity score between every possible pair of recommendations")
            print("           for a given user. Higher means more diverse.")
            print("Freshness:   Average popularity rank of recommended items. Higher means more novel.")

    def Endorsement(self, ml, USER=85, k=10):

        for algo in self.algorithms:
            print("\nUsing recommender ", algo.Label())

            print("\nBuilding recommendation model...")
            trainSet = self.dataset.EntireTraining()
            algo.Method().fit(trainSet)

            print("Computing recommendations...")
            testSet = self.dataset.OppositeTesting(USER)

            predictions = algo.Method().test(testSet)

            recommendations = []

            print("\nWe recommend:")
            for userID, movieID, actualRating, estimatedRating, _ in predictions:
                intMovieID = int(movieID)
                recommendations.append((intMovieID, estimatedRating))

            recommendations.sort(key=lambda x: x[1], reverse=True)

            for ratings in recommendations[:5]:
                print(ml.filmLabel(ratings[0]), ratings[1])

def Amount():
    ml = Films()
    print("Loading movie ratings...")
    data = ml.download()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.celebrity()
    return (ml, data, rankings)




"""
#######################################   DashBoard    ###########################################
"""





np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = Amount()

# Construct an referee to, you know, evaluate them
evaluator = referee(evaluationData, rankings)

# User-based KNN
UserKNN = KNNBasic(sim_options = {'name': 'cosine', 'user_based': True})
evaluator.SumMethod(UserKNN, "User KNN")

# Item-based KNN
ItemKNN = KNNBasic(sim_options = {'name': 'cosine', 'user_based': False})
evaluator.SumMethod(ItemKNN, "Item KNN")

# Just make random recommendations
Random = NormalPredictor()
evaluator.SumMethod(Random, "Random")

# Fight!
evaluator.Appraise(False)

evaluator.Endorsement(ml)
