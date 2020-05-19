using System.Collections.Generic;
using System.Linq;
using GingerbreadAI.NLP.Word2Vec.AnalysisFunctions;
using GingerbreadAI.NLP.Word2Vec.Embeddings;
using Xunit;

namespace GingerbreadAI.NLP.Word2Vec.Test.AnalysisFunctions
{
    public class DBSCANShould
    {
        [Theory]
        [MemberData(nameof(GetCorrectlyGetClusterLabelsForWordsTestData))]
        public void CorrectlyGetClusterLabels(List<WordEmbedding> wordEmbeddings, (string[] elements, bool isNoise)[] expectedGroups)
        {
            var labels = DBSCAN.GetLabelClusterIndexMap(
                wordEmbeddings,
                3,
                2,
                concurrentThreads: 1
            );

            foreach (var (elements, isNoise) in expectedGroups)
            {
                var groupLabel = isNoise
                    ? -1
                    : labels.First(l => l.Key == elements[0]).Value;
                foreach (var label in labels.Where(l => elements.Contains(l.Key)))
                {
                    Assert.Equal(groupLabel, label.Value);
                }
            }
        }

        [Theory]
        [MemberData(nameof(GetCorrectlyGetClusterLabelsForWordsTestData))]
        public void CorrectlyGetClusterLabelsConcurrently(List<WordEmbedding> wordVectorWeights, (string[] elements, bool isNoise)[] expectedGroups)
        {
            var labels = DBSCAN.GetLabelClusterIndexMap(
                wordVectorWeights,
                3,
                2
            );

            foreach (var (elements, isNoise) in expectedGroups)
            {
                var groupLabel = isNoise
                    ? -1
                    : labels.First(l => l.Key == elements[0]).Value;
                foreach (var label in labels.Where(l => elements.Contains(l.Key)))
                {
                    Assert.Equal(groupLabel, label.Value);
                }
            }
        }

        public static IEnumerable<object[]> GetCorrectlyGetClusterLabelsForWordsTestData()
        {
            yield return new object[]
            {
                new List<WordEmbedding>
                {
                    new WordEmbedding("a", new[] {1d, 1d}),
                    new WordEmbedding("b", new[] {2d, 2d}),
                    new WordEmbedding("c", new[] {3d, 3d}),
                    new WordEmbedding("d", new[] {4d, 4d}),
                    new WordEmbedding("e", new[] {5d, 5d}),
                    new WordEmbedding("f", new[] {6d, 6d})
                },
                new (string[] elements, bool isNoise)[]
                {
                    (new [] {"a", "b", "c", "d", "e", "f"}, false)
                }
            };
            yield return new object[]
            {
                new List<WordEmbedding>
                {
                    new WordEmbedding("a", new[] {1d, 1d}),
                    new WordEmbedding("b", new[] {-2d, -2d}),
                    new WordEmbedding("c", new[] {3d, 3d}),
                    new WordEmbedding("d", new[] {-4d, -4d}),
                    new WordEmbedding("e", new[] {5d, 5d}),
                    new WordEmbedding("f", new[] {-6d, -6d}),
                    new WordEmbedding("g", new[] {7d, 7d}),
                    new WordEmbedding("h", new[] {-8d, -8d}),
                    new WordEmbedding("i", new[] {9d, 9d}),
                    new WordEmbedding("j", new[] {-10d, -10d}),
                    new WordEmbedding("k", new[] {11d, 11d}),
                    new WordEmbedding("l", new[] {-12d, -12d}),
                    new WordEmbedding("m", new[] {13d, 13d}),
                    new WordEmbedding("n", new[] {-14d, -14d}),
                    new WordEmbedding("o", new[] {15d, 15d}),
                    new WordEmbedding("p", new[] {-16d, -16d}),
                    new WordEmbedding("q", new[] {17d, 17d}),
                    new WordEmbedding("r", new[] {-18d, -18d}),
                    new WordEmbedding("s", new[] {19d, 19d}),
                    new WordEmbedding("t", new[] {-20d, -20d}),
                    new WordEmbedding("u", new[] {21d, 21d}),
                    new WordEmbedding("v", new[] {-22d, -22d}),
                    new WordEmbedding("w", new[] {23d, 23d}),
                    new WordEmbedding("x", new[] {-24d, -24d}),
                },
                new (string[] elements, bool isNoise)[]
                {
                    (new [] { "a", "c", "e", "g", "i", "k", "m", "o", "q", "s", "u", "w"}, false),
                    (new [] { "b", "d", "f", "h", "j", "l", "n", "p", "r", "t", "v", "x"}, false)
                }
            };
            yield return new object[]
            {
                new List<WordEmbedding>
                {
                    new WordEmbedding("a", new[] {1d, 2d}),
                    new WordEmbedding("b", new[] {2d, 2d}),
                    new WordEmbedding("c", new[] {2d, 3d}),
                    new WordEmbedding("d", new[] {8d, 7d}),
                    new WordEmbedding("e", new[] {8d, 8d}),
                    new WordEmbedding("f", new[] {25d, 80d})
                },
                new (string[] elements, bool isNoise)[]
                {
                    (new [] {"a", "b", "c"}, false),
                    (new [] {"d", "e"}, false),
                    (new [] {"f"}, true),
                }
            };
            yield return new object[]
            {
                new List<WordEmbedding>
                {
                    new WordEmbedding("a", new[] {0d, 0d}),
                    new WordEmbedding("d", new[] {10d, 10d}),
                    new WordEmbedding("e", new[] {11d, 11d}),
                    new WordEmbedding("b", new[] {2d, 2d}),
                    new WordEmbedding("f", new[] {25d, 80d}),
                    new WordEmbedding("c", new[] {4d, 4d}),
                },
                new (string[] elements, bool isNoise)[]
                {
                    (new [] {"a", "b", "c"}, false),
                    (new [] {"d", "e"}, false),
                    (new [] {"f"}, true),
                }
            };
            yield return new object[]
            {
                new List<WordEmbedding>
                {
                    new WordEmbedding("a", new[] {25d, 80d}),
                    new WordEmbedding("b", new[] {8d, 8d}),
                    new WordEmbedding("c", new[] {8d, 7d}),
                    new WordEmbedding("d", new[] {2d, 3d}),
                    new WordEmbedding("e", new[] {2d, 2d}),
                    new WordEmbedding("f", new[] {1d, 2d}),
                },
                new (string[] elements, bool isNoise)[]
                {
                    (new [] {"a"}, true),
                    (new [] {"b", "c"}, false),
                    (new [] {"d", "e", "f"}, false),
                }
            };
        }
    }
}
