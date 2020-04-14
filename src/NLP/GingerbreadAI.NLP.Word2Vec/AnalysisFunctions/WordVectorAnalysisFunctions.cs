using System;
using System.Collections.Generic;
using System.Linq;

namespace GingerbreadAI.NLP.Word2Vec.AnalysisFunctions
{
    public static class WordVectorAnalysisFunctions
    {
        public static List<string> GetMostSimilarWords(string word, List<(string word, List<double> vector)> wordVectors, int topn = 10)
        {
            var wordVector = wordVectors.First(wv => wv.word == word);

            var otherWordCosineSimilarity = new List<(string otherWord, double cosineSimilarity)>();
            foreach (var otherWordVector in wordVectors.Where(wv => wv.word != word))
            {
                otherWordCosineSimilarity.Add((otherWordVector.word, CalculateCosineSimilarity(wordVector.vector.ToArray(), otherWordVector.vector.ToArray())));
            }

            var orderedOtherWordCosineSimilarity = otherWordCosineSimilarity
                .OrderByDescending(owcs => owcs.cosineSimilarity)
                .Select(owcs => owcs.otherWord);
            return orderedOtherWordCosineSimilarity.Take(topn).ToList();
        }

        private static double CalculateCosineSimilarity(double[] vectorA, double[] vectorB)
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
