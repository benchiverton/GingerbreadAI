using System;
using System.Collections.Generic;
using System.Linq;
using GingerbreadAI.NLP.Word2Vec.DistanceFunctions;
using GingerbreadAI.NLP.Word2Vec.Embeddings;

namespace GingerbreadAI.NLP.Word2Vec.AnalysisFunctions;

/// <summary>
/// Implementation of k-means clustering.
/// More info: https://en.wikipedia.org/wiki/K-means_clustering
/// </summary>
public class KMeans
{
    private readonly IEnumerable<IEmbedding> _embeddings;

    public KMeans(IEnumerable<IEmbedding> embeddings) => _embeddings = embeddings;

    public Dictionary<int, double[]> Clusters { get; private set; }
    public Dictionary<string, int> LabelClusterMap { get; private set; }

    /// <summary>
    /// Calculates and Sets the LabelClusterMap, which maps each label to its cluster.
    /// </summary>
    public void CalculateLabelClusterMap(
        int numberOfClusters = 10,
        int numberOfRuns = 10,
        int maximumIterations = 300,
        double tolerance = 1e-4)
    {
        var embeddingsAsArray = _embeddings.ToArray();

        var bestInertia = double.MaxValue;
        var clusterIds = new int[numberOfClusters];
        for (var i = 0; i < numberOfClusters; i++)
        {
            clusterIds[i] = i;
        }

        for (var i = 0; i < numberOfRuns; i++)
        {
            var clusters = GetInitialClusters(embeddingsAsArray, clusterIds);
            var solution = CalculateSolution(clusters, embeddingsAsArray);
            Dictionary<int, double[]> oldClusters;
            var iteration = 0;

            do
            {
                iteration++;

                oldClusters = clusters;
                clusters = RecalculateClusters(solution, embeddingsAsArray, clusterIds);
                solution = CalculateSolution(clusters, embeddingsAsArray);
            } while (iteration <= maximumIterations && !DeclareConvergence(oldClusters, clusters, tolerance));

            var inertia = CalculateInertia(clusters, solution, embeddingsAsArray);
            if (inertia < bestInertia)
            {
                Clusters = clusters;
                bestInertia = inertia;
                LabelClusterMap = solution;
            }
        }
    }

    /// <summary>
    /// Distortion is the sum of the square of the distances of each point from its cluster.
    /// This is often used to help find the optimal number of clusters.
    /// </summary>
    public double CalculateDistortion()
    {
        var distortion = 0d;
        foreach (var embedding in _embeddings)
        {
            var clusterEmbedding = Clusters[LabelClusterMap[embedding.Label]];
            for (var i = 0; i < embedding.Vector.Length; i++)
            {
                distortion += Math.Pow(embedding.Vector[i] - clusterEmbedding[i], 2);
            }
        }
        return distortion;
    }

    private static Dictionary<int, double[]> GetInitialClusters(IEmbedding[] embeddings, int[] clusterIds)
    {
        var random = new Random();
        var initialClusters = new Dictionary<int, double[]>();

        foreach (var clusterId in clusterIds)
        {
            initialClusters.Add(clusterId, embeddings[random.Next() % embeddings.Length].Vector);
        }

        return initialClusters;
    }

    private static Dictionary<string, int> CalculateSolution(Dictionary<int, double[]> clusters, IEmbedding[] embeddings)
    {
        var distanceFunction = DistanceFunctionResolver.ResolveDistanceFunction(DistanceFunctionType.Euclidean);
        var solution = new Dictionary<string, int>();

        foreach (var embedding in embeddings)
        {
            var closestClusterIndex = -1;
            var distanceFromClosestCluster = double.MaxValue;
            foreach (var cluster in clusters)
            {
                var distanceFromCluster = distanceFunction.Invoke(cluster.Value, embedding.Vector);
                if (distanceFromCluster < distanceFromClosestCluster)
                {
                    closestClusterIndex = cluster.Key;
                    distanceFromClosestCluster = distanceFromCluster;
                }
            }
            solution.Add(embedding.Label, closestClusterIndex);
        }

        return solution;
    }

    private static Dictionary<int, double[]> RecalculateClusters(Dictionary<string, int> solution, IEmbedding[] embeddings, int[] clusterIds)
    {
        var dimensions = embeddings[0].Vector.Length;
        var clusters = clusterIds.ToDictionary(cid => cid, cid => new double[dimensions]);
        var elementsInClusters = clusterIds.ToDictionary(cid => cid, cid => 0);

        foreach (var embedding in embeddings)
        {
            var clusterId = solution[embedding.Label];
            clusters[clusterId] = clusters[clusterId].Zip(embedding.Vector, (x, y) => x + y).ToArray();
            elementsInClusters[clusterId]++;
        }

        foreach (var (clusterId, vector) in clusters)
        {
            var elementsInCluster = elementsInClusters[clusterId];
            for (var i = 0; i < vector.Length; i++)
            {
                vector[i] /= elementsInCluster;
            }
        }

        return clusters;
    }

    private static bool DeclareConvergence(Dictionary<int, double[]> oldClusters, Dictionary<int, double[]> newClusters, double tolerance)
    {
        var distanceFunction = DistanceFunctionResolver.ResolveDistanceFunction(DistanceFunctionType.Euclidean);

        foreach (var (clusterId, oldVector) in oldClusters)
        {
            if (distanceFunction.Invoke(oldVector, newClusters[clusterId]) > tolerance)
            {
                return false;
            }
        }

        return true;
    }

    private static double CalculateInertia(Dictionary<int, double[]> clusters, Dictionary<string, int> solution, IEmbedding[] embeddings)
    {
        var distanceFunction = DistanceFunctionResolver.ResolveDistanceFunction(DistanceFunctionType.Euclidean);
        var inertia = 0d;

        foreach (var embedding in embeddings)
        {
            var clusterCenter = clusters[solution[embedding.Label]];
            inertia += distanceFunction.Invoke(clusterCenter, embedding.Vector);
        }

        return inertia;
    }
}
