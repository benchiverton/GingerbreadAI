using System;
using System.Collections.Generic;
using System.Linq;

namespace GingerbreadAI.NLP.Word2Vec.AnalysisFunctions
{
    public static class WordVectorAnalysisFunctions
    {
        /// <summary>
        /// Returns the topn most similar words to the one given.
        /// Similarity is calculated using cosine similarity.
        /// </summary>
        public static IEnumerable<(string word, double similarity)> GetMostSimilarWords(string word, IEnumerable<(string word, List<double> vector)> wordVectors, int topn = 10)
        {
            var wordVectorArray = wordVectors.ToArray();
            var wordVector = wordVectorArray.First(wv => wv.word == word);

            return wordVectorArray.Where(wv => wv.word != word)
                .Select(otherWordVector => (otherWordVector.word, CalculateCosineSimilarity(wordVector.vector.ToArray(), otherWordVector.vector.ToArray())))
                .OrderByDescending(owcs => owcs.Item2)
                .Take(topn);
        }

        internal static double CalculateCosineSimilarity(double[] vectorA, double[] vectorB)
        {
            if (vectorA.Length != vectorB.Length)
            {
                throw new Exception($"Can't calculate the cosine similarity as the length of vector A ({vectorA.Length}) is different to the length of vector B ({vectorB.Length}).");
            }

            var numerator = 0d;
            var denominatorA = 0d;
            var denominatorB = 0d;
            for (var i = 0; i < vectorA.Length; i++)
            {
                numerator += vectorA[i] * vectorB[i];
                denominatorA += Math.Pow(vectorA[i], 2);
                denominatorB += Math.Pow(vectorB[i], 2);
            }

            return numerator / (Math.Sqrt(denominatorA) * Math.Sqrt(denominatorB));
        }
    }
}
