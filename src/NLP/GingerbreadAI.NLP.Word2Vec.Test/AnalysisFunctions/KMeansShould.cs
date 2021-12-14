using System.Collections.Generic;
using System.Linq;
using GingerbreadAI.NLP.Word2Vec.AnalysisFunctions;
using GingerbreadAI.NLP.Word2Vec.Embeddings;
using Xunit;

namespace GingerbreadAI.NLP.Word2Vec.Test.AnalysisFunctions;

public class KMeansShould
{
    [Theory]
    [MemberData(nameof(GetPutXElementsInXClustersData))]
    public void PutXElementsInXClusters(int x)
    {
        var embeddings = new List<WordEmbedding>();
        for (var i = 0; i < x; i++)
        {
            var embedding = new double[x];
            embedding[i] = 1d;
            embeddings.Add(new WordEmbedding(i.ToString(), embedding));
        }
        var kMeans = new KMeans(embeddings);

        kMeans.CalculateLabelClusterMap(numberOfClusters: x);

        Assert.Equal(x, kMeans.LabelClusterMap.Keys.Distinct().Count());
    }

    [Theory]
    [MemberData(nameof(GetCorrectlyAssignClustersData))]
    public void CorrectlyAssignClusters(List<IEmbedding> embeddings, int numberOfClusters, List<List<string>> desiredClusters)
    {
        var kMeans = new KMeans(embeddings);

        kMeans.CalculateLabelClusterMap(numberOfClusters: numberOfClusters);

        var clusters = kMeans.LabelClusterMap.GroupBy(lcm => lcm.Value);
        foreach (var cluster in clusters)
        {
            var elements = cluster.ToArray();
            var desiredCluster = desiredClusters.First(dc => dc.Contains(elements[0].Key));

            // assert desired and actual cluster contain the same elements
            Assert.Equal(desiredCluster.Count, elements.Length);
            foreach (var element in elements)
            {
                Assert.Contains(desiredCluster, x => x == element.Key);
            }
        }
    }

    public static IEnumerable<object[]> GetPutXElementsInXClustersData()
    {
        yield return new object[] { 1 };
        yield return new object[] { 2 };
        yield return new object[] { 5 };
        yield return new object[] { 10 };
    }

    public static IEnumerable<object[]> GetCorrectlyAssignClustersData()
    {
        yield return new object[] { new List<IEmbedding>
            {
                new WordEmbedding("a", new [] {1d, 1d, 1d} ),
                new WordEmbedding("b", new [] {2d, 2d, 2d} ),
                new WordEmbedding("c", new [] {3d, 3d, 3d} )
            }, 3, new List<List<string>>
            {
                new List<string> { "a" },
                new List<string> { "b" },
                new List<string> { "c" }
            } };

        yield return new object[] { new List<IEmbedding>
            {
                new WordEmbedding("a", new [] {1d, 1d, 1d} ),
                new WordEmbedding("b", new [] {2d, 2d, 2d} ),
                new WordEmbedding("c", new [] {3d, 3d, 3d} )
            }, 1, new List<List<string>>
            {
                new List<string> { "a", "b", "c" }
            } };

        yield return new object[] { new List<IEmbedding>
            {
                new WordEmbedding("a", new [] {-1d, -1d, -1d} ),
                new WordEmbedding("b", new [] {2d, 2d, 2d} ),
                new WordEmbedding("c", new [] {3d, 3d, 3d} )
            }, 2, new List<List<string>>
            {
                new List<string> { "a" },
                new List<string> { "b", "c" }
            } };
    }
}
