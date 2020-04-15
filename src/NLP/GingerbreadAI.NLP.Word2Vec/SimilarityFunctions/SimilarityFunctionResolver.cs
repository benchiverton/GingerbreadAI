using System;

namespace GingerbreadAI.NLP.Word2Vec.SimilarityFunctions
{
    public static class SimilarityFunctionResolver
    {
        public static Func<double[], double[], double> ResolveSimilarityFunction(SimilarityFunctionType similarityFunctionType)
        {
            return similarityFunctionType switch
            {
                SimilarityFunctionType.Cosine => CalculateCosineSimilarity,
                _ => throw new ArgumentOutOfRangeException(
                    nameof(similarityFunctionType),
                    similarityFunctionType,
                    "This similarity function type is not yet supported. Please use a different similarity function type.")
            };
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
