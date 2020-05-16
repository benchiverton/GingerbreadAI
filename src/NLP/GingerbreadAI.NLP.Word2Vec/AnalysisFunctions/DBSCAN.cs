using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using GingerbreadAI.NLP.Word2Vec.DistanceFunctions;
using GingerbreadAI.NLP.Word2Vec.Embeddings;

namespace GingerbreadAI.NLP.Word2Vec.AnalysisFunctions
{
    /// <summary>
    /// Implementation of Density-Based Spatial Clustering Of Applications with Noise for data clustering.
    /// More info: https://en.wikipedia.org/wiki/DBSCAN
    /// </summary>
    public static class DBSCAN
    {
        public static Dictionary<string, int> GetLabelClusterIndexMap(
            IEnumerable<IEmbedding> embeddings,
            double epsilon = 0.5,
            int minimumSamples = 5,
            DistanceFunctionType distanceFunctionType = DistanceFunctionType.Euclidean,
            int concurrentThreads = 4)
        {
            var embeddingsList = embeddings.ToList();

            var distanceFunction = DistanceFunctionResolver.ResolveDistanceFunction(distanceFunctionType);

            var clusterLabels = new ConcurrentDictionary<string, int>();
            var clusterRelationships = new ConcurrentBag<ConcurrentBag<int>>();
            var clusterIndex = 0;
            var sampleSize = (int)Math.Ceiling((double)embeddingsList.Count / concurrentThreads);

            Parallel.For(0, concurrentThreads, threadIndex =>
            {
                foreach (var embedding in embeddingsList.Skip(threadIndex * sampleSize).Take(sampleSize))
                {
                    if (clusterLabels.ContainsKey(embedding.Label))
                    {
                        continue;
                    }

                    var neighbors = GetNeighborsAndWeight(
                        embedding,
                        embeddingsList,
                        distanceFunction,
                        epsilon);

                    if (neighbors.Count < minimumSamples)
                    {
                        clusterLabels.AddOrUpdate(
                            embedding.Label,
                            -1,
                            (key, existingClusterIndex) => existingClusterIndex);
                        continue;
                    }

                    var localClusterIndex = clusterIndex++;
                    clusterLabels.AddOrUpdate(
                        embedding.Label,
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
                        if (clusterLabels.TryGetValue(currentNeighbor.Label, out var existingClusterId))
                        {
                            if (existingClusterId != -1 && existingClusterId != localClusterIndex)
                            {
                                clusterRelationships.First(r => r.Contains(existingClusterId)).Add(localClusterIndex);
                            }
                            clusterLabels[currentNeighbor.Label] = localClusterIndex;
                            continue;
                        }

                        clusterLabels.AddOrUpdate(
                            currentNeighbor.Label,
                            localClusterIndex,
                            (key, existingClusterIndex) =>
                            {
                                clusterRelationships.First(r => r.Contains(existingClusterIndex)).Add(localClusterIndex);
                                return localClusterIndex;
                            });

                        var currentNeighborsNeighbors = GetNeighborsAndWeight(
                            currentNeighbor,
                            embeddingsList,
                            distanceFunction,
                            epsilon);

                        if (currentNeighborsNeighbors.Count >= minimumSamples)
                        {
                            neighbors = neighbors.Union(currentNeighborsNeighbors).ToList();
                        }
                    }
                }
            });

            var clusterIndexMap = GetClusterIndexMap(clusterRelationships);

            return clusterLabels.ToDictionary(
                x => x.Key,
                x => clusterIndexMap[x.Value]);
        }

        private static List<IEmbedding> GetNeighborsAndWeight(
            IEmbedding currentEmbedding,
            IEnumerable<IEmbedding> embeddings,
            Func<double[], double[], double> distanceFunction,
            double epsilon)
        {
            var neighbors = new List<IEmbedding>();
            foreach (var embedding in embeddings)
            {
                var distance = distanceFunction.Invoke(currentEmbedding.Vector, embedding.Vector);
                if (distance < epsilon)
                {
                    neighbors.Add(embedding);
                }
            }
            return neighbors;
        }

        /// <summary>
        /// Gets the Cluster Index Map, mapping related clusters to the same cluster index.
        /// eg: (0,1) (1,2) (3,4) => [0:0] [1:0] [2:0] [3:1] [4:1].
        /// </summary>
        private static Dictionary<int, int> GetClusterIndexMap(ConcurrentBag<ConcurrentBag<int>> clusterRelationships)
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
