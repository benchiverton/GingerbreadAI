using System;
using GingerbreadAI.NLP.Word2Vec.SimilarityFunctions;

namespace GingerbreadAI.NLP.Word2Vec.DistanceFunctions
{
    public static class DistanceFunctionResolver
    {
        public static Func<double[], double[], double> ResolveDistanceFunction(
            DistanceFunctionType distanceFunctionType)
        {
            return distanceFunctionType switch
            {
                DistanceFunctionType.Euclidean => CalculateEuclideanDistance,
                DistanceFunctionType.Cosine => CalculateCosineDistance,
                _ => throw new ArgumentOutOfRangeException(
                    nameof(distanceFunctionType),
                    distanceFunctionType,
                    "This distance function type is not yet supported. Please use a different distance function type.")
            };
        }

        private static double CalculateEuclideanDistance(double[] vectorA, double[] vectorB)
        {
            var sumOfSquareDistances = 0d;

            for (var i = 0; i < vectorA.Length; i++)
            {
                sumOfSquareDistances += Math.Pow(vectorA[i] - vectorB[i], 2);
            }

            return Math.Sqrt(sumOfSquareDistances);
        }

        private static double CalculateCosineDistance(double[] vectorA, double[] vectorB)
        {
            return 1d - SimilarityFunctionResolver.ResolveSimilarityFunction(SimilarityFunctionType.Cosine)
                       .Invoke(vectorA, vectorB);
        }
    }
}
