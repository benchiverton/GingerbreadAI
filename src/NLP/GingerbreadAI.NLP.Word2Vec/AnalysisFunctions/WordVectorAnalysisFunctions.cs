﻿using System;
using System.Collections.Generic;
using System.Linq;
using GingerbreadAI.NLP.Word2Vec.DistanceFunctions;
using GingerbreadAI.NLP.Word2Vec.SimilarityFunctions;

namespace GingerbreadAI.NLP.Word2Vec.AnalysisFunctions
{
    public static class WordVectorAnalysisFunctions
    {
        /// <summary>
        /// Returns the topn most similar words to the one given.
        /// </summary>
        public static IEnumerable<(string word, double similarity)> GetMostSimilarWords(
            string word, 
            IEnumerable<(string word, double[] vector)> wordVectors, 
            int topn = 10, 
            SimilarityFunctionType similarityFunctionType = SimilarityFunctionType.Cosine)
        {
            var similarityFunction = SimilarityFunctionResolver.ResolveSimilarityFunction(similarityFunctionType);

            var wordVectorArray = wordVectors.ToArray();
            var wordVector = wordVectorArray.First(wv => wv.word == word);

            return wordVectorArray.Where(wv => wv.word != word)
                .Select(otherWordVector => (otherWordVector.word, similarityFunction.Invoke(wordVector.vector.ToArray(), otherWordVector.vector.ToArray())))
                .OrderByDescending(owcs => owcs.Item2)
                .Take(topn);
        }

        /// <summary>
        /// Calculate cluster labels from Density-Based Spatial Clustering Of Applications with Noise (DBSCAN).
        /// </summary>
        public static Dictionary<string, int> GetClusterLabels(
            List<(string word, double[] vector)> wordVectors,
            double epsilon = 0.5,
            int minimumSamples = 5,
            DistanceFunctionType distanceFunctionType = DistanceFunctionType.Euclidean)
        {
            var clusterLabels = new Dictionary<string, int>();
            var distanceFunction = DistanceFunctionResolver.ResolveDistanceFunction(distanceFunctionType);

            var clusterIndex = -1;
            foreach (var wordVectorWeight in wordVectors)
            {
                if (clusterLabels.ContainsKey(wordVectorWeight.word))
                {
                    continue;
                }
                clusterLabels.Add(wordVectorWeight.word, -1);

                var neighbors = GetNeighborsAndWeight(wordVectorWeight, wordVectors, distanceFunction, epsilon);

                if (neighbors.totalWeight < minimumSamples)
                {
                    continue;
                }

                clusterIndex += 1;
                clusterLabels[wordVectorWeight.word] = clusterIndex;

                for (var i = 0; i < neighbors.neighbors.Count; i++)
                {
                    var currentNeighbor = neighbors.neighbors[i];
                    if (clusterLabels.ContainsKey(currentNeighbor.word))
                    {
                        if (clusterLabels[currentNeighbor.word] == -1)
                        {
                            clusterLabels[currentNeighbor.word] = clusterIndex;
                        }
                        continue;
                    }

                    clusterLabels.Add(currentNeighbor.word, clusterIndex);

                    var currentNeighborsNeighbors = GetNeighborsAndWeight(
                        currentNeighbor,
                        wordVectors,
                        distanceFunction,
                        epsilon);

                    if (currentNeighborsNeighbors.totalWeight >= minimumSamples)
                    {
                        neighbors.neighbors = neighbors.neighbors.Union(currentNeighborsNeighbors.neighbors).ToList();
                    }
                }
            }

            return clusterLabels;
        }

        private static (List<(string word, double[] vector)> neighbors, int totalWeight) GetNeighborsAndWeight(
            (string word, double[] vector) currentWordVectorWeight,
            IEnumerable<(string word, double[] vector)> wordVectorWeights,
            Func<double[], double[], double> distanceFunction,
            double epsilon)
        {
            var neighbors = new List<(string word, double[] vector)>();
            var totalWeight = 0;
            foreach (var wordVector in wordVectorWeights)
            {
                var distance = distanceFunction.Invoke(currentWordVectorWeight.vector, wordVector.vector);
                if (distance < epsilon)
                {
                    neighbors.Add(wordVector);
                    totalWeight += 1;
                }
            }

            // remove current word from neighbors
            neighbors.Remove(currentWordVectorWeight);

            return (neighbors, totalWeight);
        }
    }
}