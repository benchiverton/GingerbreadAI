using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
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
            DistanceFunctionType distanceFunctionType = DistanceFunctionType.Euclidean,
            int concurrentThreads = 4)
        {
            var distanceFunction = DistanceFunctionResolver.ResolveDistanceFunction(distanceFunctionType);

            var clusterLabels = new ConcurrentDictionary<string, int>();
            var clusterIndexMap = new Dictionary<int, int> { { -1, -1 } };
            var clusterIndex = 0;
            var sampleSize = (int)Math.Ceiling((double)wordVectors.Count / concurrentThreads);

            Parallel.For(0, concurrentThreads, threadIndex =>
            {
                foreach (var wordVector in wordVectors.Skip(threadIndex * sampleSize).Take(sampleSize))
                {
                    if (clusterLabels.ContainsKey(wordVector.word))
                    {
                        continue;
                    }

                    var neighbors = GetNeighborsAndWeight(
                        wordVector,
                        wordVectors,
                        distanceFunction,
                        epsilon);

                    if (neighbors.Count < minimumSamples)
                    {
                        clusterLabels.AddOrUpdate(
                            wordVector.word,
                            -1,
                            (key, oldValue) => oldValue);
                        continue;
                    }

                    var localClusterIndex = -999;
                    clusterLabels.AddOrUpdate(
                        wordVector.word,
                        (key =>
                        {
                            localClusterIndex = clusterIndex++;
                            clusterIndexMap.Add(localClusterIndex, localClusterIndex);
                            return localClusterIndex;
                        }),
                        (key, existingClusterIndex) =>
                        {
                            localClusterIndex = clusterIndexMap[existingClusterIndex];
                            return localClusterIndex;
                        });

                    for (var i = 0; i < neighbors.Count; i++)
                    {
                        var currentNeighbor = neighbors[i];
                        if (clusterLabels.TryGetValue(currentNeighbor.word, out var existingClusterId))
                        {
                            if (existingClusterId != -1)
                            {
                                UpdateClusterIndexMap(localClusterIndex, existingClusterId, clusterIndexMap);
                                localClusterIndex = clusterIndexMap[existingClusterId];
                            }
                            clusterLabels[currentNeighbor.word] = localClusterIndex;
                            continue;
                        }

                        clusterLabels.AddOrUpdate(
                            currentNeighbor.word,
                            localClusterIndex,
                            (key, existingClusterIndex) =>
                            {
                                UpdateClusterIndexMap(localClusterIndex, existingClusterIndex, clusterIndexMap);
                                return localClusterIndex;
                            });

                        var currentNeighborsNeighbors = GetNeighborsAndWeight(
                            currentNeighbor,
                            wordVectors,
                            distanceFunction,
                            epsilon);

                        if (currentNeighborsNeighbors.Count >= minimumSamples)
                        {
                            neighbors = neighbors.Union(currentNeighborsNeighbors).ToList();
                        }
                    }
                }
            });

            FlattenLabelClusterMap(clusterIndexMap);

            return clusterLabels.ToDictionary(
                x => x.Key,
                x => clusterIndexMap[x.Value]);
        }

        private static List<(string word, double[] vector)> GetNeighborsAndWeight(
            (string word, double[] vector) currentWordVectorWeight,
            IEnumerable<(string word, double[] vector)> wordVectorWeights,
            Func<double[], double[], double> distanceFunction,
            double epsilon)
        {
            var neighbors = new List<(string word, double[] vector)>();
            foreach (var wordVector in wordVectorWeights)
            {
                var distance = distanceFunction.Invoke(currentWordVectorWeight.vector, wordVector.vector);
                if (distance < epsilon)
                {
                    neighbors.Add(wordVector);
                }
            }

            return neighbors;
        }

        /// <summary>
        /// Updates the label cluster map for the higher index to the lower index.
        /// </summary>
        private static void UpdateClusterIndexMap(int localClusterIndex, int existingClusterIndex, IDictionary<int, int> labelClusterMap)
        {
            if (existingClusterIndex == localClusterIndex) return;

            if (existingClusterIndex < localClusterIndex)
            {
                labelClusterMap[localClusterIndex] = existingClusterIndex;
            }
            else if (localClusterIndex < existingClusterIndex)
            {
                labelClusterMap[existingClusterIndex] = localClusterIndex;
            }
        }

        /// <summary>
        /// Flattens the Label Cluster Map so that each label maps to the original cluster index.
        /// eg: (0,0) (1,0) (2,1) => (0,0) (1,0) (2,0).
        /// </summary>
        private static void FlattenLabelClusterMap(IDictionary<int, int> labelClusterMap)
        {
            foreach (var label in labelClusterMap.Keys.ToArray())
            {
                var i = labelClusterMap[label];
                while (labelClusterMap[i] != i)
                {
                    i = labelClusterMap[i];
                }

                labelClusterMap[label] = i;
            }
        }
    }
}
