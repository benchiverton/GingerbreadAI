using System.Collections.Generic;
using System.Linq;
using GingerbreadAI.NLP.Word2Vec.AnalysisFunctions;
using Xunit;

namespace GingerbreadAI.NLP.Word2Vec.Test.AnalysisFunctions
{
    public class WordVectorAnalysisFunctionsShould
    {
        [Fact]
        public void CorrectlyGetMostSimilarWords()
        {
            var wordVectors = new (string word, double[] vector) []
            {
                ("target", new [] {1d, 1d}),
                ("far", new [] {-1d, -1d}),
                ("close-ish", new [] {-1d, 1d}),
                ("same", new [] {1d, 1d}),
            };

            var orderedWords = WordVectorAnalysisFunctions.GetMostSimilarWords("target", wordVectors, 3).ToArray();
            Assert.Equal(1d, orderedWords[0].similarity, 8);
            Assert.Equal("same", orderedWords[0].word);
            Assert.Equal(0d, orderedWords[1].similarity, 8);
            Assert.Equal("close-ish", orderedWords[1].word);
            Assert.Equal(-1d, orderedWords[2].similarity, 8);
            Assert.Equal("far", orderedWords[2].word);
        }

        [Theory]
        [MemberData(nameof(GetCorrectlyGetClusterLabelsForWordsTestData))]
        public void CorrectlyGetClusterLabels(List<(string word, double[] vector)> wordVectorWeights, Dictionary<string, int> expectedResults)
        {
            var labels = WordVectorAnalysisFunctions.GetClusterLabels(
                wordVectorWeights,
                3,
                2
            );

            foreach (var (word, clusterIndex) in labels)
            {
                Assert.Equal(expectedResults[word], clusterIndex);
            }
        }

        public static IEnumerable<object[]> GetCorrectlyGetClusterLabelsForWordsTestData()
        {
            yield return new object[]
            {
                new List<(string word, double[] vector)>
                {
                    ("a", new[] {1d, 2d}),
                    ("b", new[] {2d, 2d}),
                    ("c", new[] {2d, 3d}),
                    ("d", new[] {8d, 7d}),
                    ("e", new[] {8d, 8d}),
                    ("f", new[] {25d, 80d})
                },
                new Dictionary<string, int>
                {
                    ["a"] = 0,
                    ["b"] = 0,
                    ["c"] = 0,
                    ["d"] = 1,
                    ["e"] = 1,
                    ["f"] = -1
                }
            };
            yield return new object[]
            {
                new List<(string word, double[] vector)>
                {
                    ("a", new[] {0d, 0d}),
                    ("d", new[] {10d, 10d}),
                    ("e", new[] {11d, 11d}),
                    ("b", new[] {2d, 2d}),
                    ("f", new[] {25d, 80d}),
                    ("c", new[] {4d, 4d}),
                },
                new Dictionary<string, int>
                {
                    ["a"] = 0,
                    ["b"] = 0,
                    ["c"] = 0,
                    ["d"] = 1,
                    ["e"] = 1,
                    ["f"] = -1
                }
            };
            yield return new object[]
            {
                new List<(string word, double[] vector)>
                {
                    ("f", new[] {25d, 80d}),
                    ("e", new[] {8d, 8d}),
                    ("d", new[] {8d, 7d}),
                    ("c", new[] {2d, 3d}),
                    ("b", new[] {2d, 2d}),
                    ("a", new[] {1d, 2d}),
                },
                new Dictionary<string, int>
                {
                    ["a"] = 1,
                    ["b"] = 1,
                    ["c"] = 1,
                    ["d"] = 0,
                    ["e"] = 0,
                    ["f"] = -1
                }
            };
        }
    }
}
