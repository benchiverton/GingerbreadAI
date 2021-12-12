using System.Collections.Generic;
using GingerbreadAI.NLP.Word2Vec.DistanceFunctions;
using Xunit;

namespace GingerbreadAI.NLP.Word2Vec.Test.DistanceFunctions;

public class DistanceFunctionResolverShould
{
    [Theory]
    [MemberData(nameof(GetCorrectlyCalculateEuclideanSimilarityTestCases))]
    public void CorrectlyCalculateEuclideanSimilarity(double[] vectorA, double[] vectorB, double expectedSimilarity)
    {
        var similarityFunction =
            DistanceFunctionResolver.ResolveDistanceFunction(DistanceFunctionType.Euclidean);

        var calculatedSimilarity = similarityFunction.Invoke(vectorA, vectorB);

        Assert.Equal(expectedSimilarity, calculatedSimilarity, 8);
    }

    public static IEnumerable<object[]> GetCorrectlyCalculateEuclideanSimilarityTestCases()
    {
        yield return new object[] { new[] { 0d }, new[] { 0d }, 0d };
        yield return new object[] { new[] { 0d }, new[] { 1d }, 1d };
        yield return new object[] { new[] { 0d }, new[] { -1d }, 1d };
        yield return new object[] { new[] { 0d, 0d }, new[] { 3d, 4d }, 5d };
    }

    [Theory]
    [MemberData(nameof(GetCorrectlyCalculateCosineSimilarityTestCases))]
    public void CorrectlyCalculateCosineSimilarity(double[] vectorA, double[] vectorB, double expectedSimilarity)
    {
        var similarityFunction =
            DistanceFunctionResolver.ResolveDistanceFunction(DistanceFunctionType.Cosine);

        var calculatedSimilarity = similarityFunction.Invoke(vectorA, vectorB);

        Assert.Equal(expectedSimilarity, calculatedSimilarity, 8);
    }

    public static IEnumerable<object[]> GetCorrectlyCalculateCosineSimilarityTestCases()
    {
        yield return new object[] { new[] { 1d, 1d }, new[] { 1d, 1d }, 0d };
        yield return new object[] { new[] { 1d, 1d }, new[] { -1d, 1d }, 1d };
        yield return new object[] { new[] { 1d, 1d }, new[] { -1d, -1d }, 2d };
    }
}
