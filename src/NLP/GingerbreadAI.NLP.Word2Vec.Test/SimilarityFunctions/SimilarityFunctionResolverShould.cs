using System.Collections.Generic;
using GingerbreadAI.NLP.Word2Vec.SimilarityFunctions;
using Xunit;

namespace GingerbreadAI.NLP.Word2Vec.Test.SimilarityFunctions
{
    public class SimilarityFunctionResolverShould
    {
        [Theory]
        [MemberData(nameof(GetCorrectlyCalculateCosineSimilarityTestCases))]
        public void CorrectlyCalculateCosineSimilarity(double[] vectorA, double[] vectorB, double expectedSimilarity)
        {
            var similarityFunction =
                SimilarityFunctionResolver.ResolveSimilarityFunction(SimilarityFunctionType.Cosine);

            var calculatedSimilarity = similarityFunction.Invoke(vectorA, vectorB);

            Assert.Equal(expectedSimilarity, calculatedSimilarity, 8);
        }

        public static IEnumerable<object[]> GetCorrectlyCalculateCosineSimilarityTestCases()
        {
            yield return new object[] { new[] { 1d }, new[] { 1d }, 1d };
            yield return new object[] { new[] { 1d }, new[] { -1d }, -1d };
            yield return new object[] { new[] { 1d, 1d }, new[] { 1d, 1d }, 1d };
            yield return new object[] { new[] { 1d, 1d }, new[] { 1d, -1d }, 0d };
            yield return new object[] { new[] { 1d, 1d }, new[] { -1d, -1d }, -1d };
        }
    }
}
