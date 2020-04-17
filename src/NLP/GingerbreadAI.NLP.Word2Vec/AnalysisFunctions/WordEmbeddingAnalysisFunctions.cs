using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Threading.Tasks;
using GingerbreadAI.NLP.Word2Vec.DistanceFunctions;
using GingerbreadAI.NLP.Word2Vec.SimilarityFunctions;

namespace GingerbreadAI.NLP.Word2Vec.AnalysisFunctions
{
    public static class WordEmbeddingAnalysisFunctions
    {
        /// <summary>
        /// Returns the topn most similar words to the one given.
        /// </summary>
        public static IEnumerable<(string word, double similarity)> GetMostSimilarWords(
            string word,
            IEnumerable<WordEmbedding> wordEmbeddings,
            int topn = 10,
            SimilarityFunctionType similarityFunctionType = SimilarityFunctionType.Cosine)
        {
            var similarityFunction = SimilarityFunctionResolver.ResolveSimilarityFunction(similarityFunctionType);

            var wordEmbeddingsArray = wordEmbeddings.ToArray();
            var wordEmbedding = wordEmbeddingsArray.First(we => we.Word == word);

            return wordEmbeddingsArray.Where(we => we.Word != word)
                .Select(otherWordEmbedding => (otherWordEmbedding.Word, similarityFunction.Invoke(wordEmbedding.Vector.ToArray(), otherWordEmbedding.Vector.ToArray())))
                .OrderByDescending(owcs => owcs.Item2)
                .Take(topn);
        }

        /// <summary>
        /// Calculate cluster labels from Density-Based Spatial Clustering Of Applications with Noise (DBSCAN).
        /// </summary>
        public static Dictionary<string, int> GetClusterLabels(
            List<WordEmbedding> wordEmbeddings,
            double epsilon = 0.5,
            int minimumSamples = 5,
            DistanceFunctionType distanceFunctionType = DistanceFunctionType.Euclidean,
            int concurrentThreads = 4)
        {
            var distanceFunction = DistanceFunctionResolver.ResolveDistanceFunction(distanceFunctionType);

            var clusterLabels = new ConcurrentDictionary<string, int>();
            var clusterRelationships = new ConcurrentBag<ConcurrentBag<int>>();
            var clusterIndex = 0;
            var sampleSize = (int)Math.Ceiling((double)wordEmbeddings.Count / concurrentThreads);

            Parallel.For(0, concurrentThreads, threadIndex =>
            {
                foreach (var wordEmbedding in wordEmbeddings.Skip(threadIndex * sampleSize).Take(sampleSize))
                {
                    if (clusterLabels.ContainsKey(wordEmbedding.Word))
                    {
                        continue;
                    }

                    var neighbors = GetNeighborsAndWeight(
                        wordEmbedding,
                        wordEmbeddings,
                        distanceFunction,
                        epsilon);

                    if (neighbors.Count < minimumSamples)
                    {
                        clusterLabels.AddOrUpdate(
                            wordEmbedding.Word,
                            -1,
                            (key, existingClusterIndex) => existingClusterIndex);
                        continue;
                    }

                    var localClusterIndex = clusterIndex++;
                    clusterLabels.AddOrUpdate(
                        wordEmbedding.Word,
                        (key) =>
                        {
                            clusterRelationships.Add(new ConcurrentBag<int> { localClusterIndex });
                            return localClusterIndex;
                        },
                        (key, existingClusterIndex) =>
                        {
                            clusterRelationships.First(r => r.Contains(existingClusterIndex)).Add(localClusterIndex);
                            return localClusterIndex;
                        });

                    for (var i = 0; i < neighbors.Count; i++)
                    {
                        var currentNeighbor = neighbors[i];
                        if (clusterLabels.TryGetValue(currentNeighbor.Word, out var existingClusterId))
                        {
                            if (existingClusterId != -1 && existingClusterId != localClusterIndex)
                            {
                                clusterRelationships.First(r => r.Contains(existingClusterId)).Add(localClusterIndex);
                            }
                            clusterLabels[currentNeighbor.Word] = localClusterIndex;
                            continue;
                        }

                        clusterLabels.AddOrUpdate(
                            currentNeighbor.Word,
                            localClusterIndex,
                            (key, existingClusterIndex) =>
                            {
                                clusterRelationships.First(r => r.Contains(existingClusterIndex)).Add(localClusterIndex);
                                return localClusterIndex;
                            });

                        var currentNeighborsNeighbors = GetNeighborsAndWeight(
                            currentNeighbor,
                            wordEmbeddings,
                            distanceFunction,
                            epsilon);

                        if (currentNeighborsNeighbors.Count >= minimumSamples)
                        {
                            neighbors = neighbors.Union(currentNeighborsNeighbors).ToList();
                        }
                    }
                }
            });

            var clusterMap = GetClusterMap(clusterRelationships);

            return clusterLabels.ToDictionary(
                x => x.Key,
                x => clusterMap[x.Value]);
        }

        private static List<WordEmbedding> GetNeighborsAndWeight(
            WordEmbedding currentWordEmbedding,
            IEnumerable<WordEmbedding> wordEmbeddings,
            Func<double[], double[], double> distanceFunction,
            double epsilon)
        {
            var neighbors = new List<WordEmbedding>();
            foreach (var wordEmbedding in wordEmbeddings)
            {
                var distance = distanceFunction.Invoke(currentWordEmbedding.Vector, wordEmbedding.Vector);
                if (distance < epsilon)
                {
                    neighbors.Add(wordEmbedding);
                }
            }
            return neighbors;
        }

        /// <summary>
        /// Gets the Cluster Map, mapping related clusters to the same cluster index.
        /// eg: (0,1) (1,2) (3,4) => [0:0] [1:0] [2:0] [3:1] [4:1].
        /// </summary>
        private static Dictionary<int, int> GetClusterMap(ConcurrentBag<ConcurrentBag<int>> clusterRelationships)
        {
            var processedClusterRelationships = new List<ConcurrentBag<int>>();
            var completeClusterRelationships = new List<List<int>>();
            foreach (var clusterRelationship in clusterRelationships.Where(cr => !processedClusterRelationships.Contains(cr)))
            {
                var localClusterRelationship = clusterRelationship.ToList();
                var relatedClusters = clusterRelationships.Where(cr 
                    => !processedClusterRelationships.Contains(cr) 
                       && cr.Intersect(localClusterRelationship).Count() != 0).ToList();
                while (relatedClusters.Any())
                {
                    foreach (var relatedCluster in relatedClusters)
                    {
                        localClusterRelationship = localClusterRelationship.Union(relatedCluster).ToList();
                        processedClusterRelationships.Add(relatedCluster);
                    }
                    relatedClusters = clusterRelationships.Where(cr
                        => !processedClusterRelationships.Contains(cr)
                           && cr.Intersect(localClusterRelationship).Count() != 0).ToList();
                }
                completeClusterRelationships.Add(localClusterRelationship);
                processedClusterRelationships.Add(clusterRelationship);
            }

            var clusterMap = new Dictionary<int, int> { { -1, -1 } };
            var i = 0;
            foreach (var completeClusterRelationship in completeClusterRelationships)
            {
                foreach (var cluster in completeClusterRelationship)
                {
                    clusterMap.Add(cluster, i);
                }
                i++;
            }

            return clusterMap;
        }
    }
}
