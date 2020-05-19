using System;
using System.Collections.Generic;
using System.Linq;
using GingerbreadAI.DeepLearning.Backpropagation;
using GingerbreadAI.DeepLearning.Backpropagation.ErrorFunctions;
using GingerbreadAI.DeepLearning.Backpropagation.Extensions;
using GingerbreadAI.Model.NeuralNetwork.ActivationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Extensions;
using GingerbreadAI.Model.NeuralNetwork.InitialisationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Models;
using GingerbreadAI.NLP.Word2Vec.DistanceFunctions;
using GingerbreadAI.NLP.Word2Vec.Embeddings;

namespace GingerbreadAI.NLP.Word2Vec.AnalysisFunctions
{
    /// <summary>
    /// Implementation of t-Distributed Stochastic Neighbor Embedding (t-SNE) for dimensionality reduction.
    /// More info: https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding
    /// Implementation heavily inspired by https://github.com/elki-project/elki
    /// </summary>
    public class TSNE
    {
        private const double MinPij = 1e-12;

        private const double PerplexityError = 1e-5;
        private const int PerplexityMaxIterations = 50;

        private readonly double _perplexity;
        private readonly DistanceFunctionType _distanceFunctionType;
        private readonly int _dimensions;
        private readonly int _iterations;

        public TSNE(
            int dimensions,
            double perplexity = 30,
            DistanceFunctionType distanceFunctionType = DistanceFunctionType.Euclidean,
            int iterations = 1000)
        {
            _dimensions = dimensions;
            _perplexity = perplexity;
            _distanceFunctionType = distanceFunctionType;
            _iterations = iterations;
        }

        public void ReduceDimensions(IEnumerable<IEmbedding> embeddings)
        {
            // TODO: make this work
            var embeddingsList = embeddings.ToList();
            var probabilityMatrix = CalculateProbabilityMatrix(embeddingsList, _distanceFunctionType, _perplexity);

            var inputLayer = new Layer(embeddingsList.Count, Array.Empty<Layer>(), ActivationFunctionType.Linear, InitialisationFunctionType.None, false);
            var outputLayer = new Layer(_dimensions, new [] { inputLayer }, ActivationFunctionType.Linear, InitialisationFunctionType.RandomGuassian, false);
            outputLayer.AddMomentumRecursively();
            outputLayer.Initialise(new Random());

            OptimiseSNE(probabilityMatrix, outputLayer);

            var results = new List<IEmbedding>();
            for (var i = 0; i < embeddingsList.Count; i++)
            {
                var inputNode = inputLayer.Nodes[i];
                var vector = outputLayer.Nodes.Select(outputNode => outputNode.Weights[inputNode].Value);

                embeddingsList[i].Vector = vector.ToArray();
            }
        }

        /// <summary>
        /// Perform t-SNE optimisation.
        /// </summary>
        private void OptimiseSNE(
            Dictionary<string, Dictionary<string, double>> probabilityMatrix,
            Layer neuralNetwork)
        {
            // targets = 0 as KL(P||Q) >= 0 for all P,Q
            var targets = new double[_dimensions];

            for (var i = 0; i < _iterations; i++)
            {
                var similarityMatrix = CalculateSimilarityMatrix(probabilityMatrix, neuralNetwork);

                var inputs = new List<double>();
                foreach (var (labelI, probabilitiesI) in probabilityMatrix)
                {
                    var inputI = 0d;
                    foreach (var (labelJ, probabilityIJ) in probabilitiesI.Where(pi => pi.Key != labelI))
                    {
                        inputI += probabilityIJ * Math.Log(probabilityIJ / similarityMatrix[labelI][labelJ]);
                    }

                    inputs.Add(inputI);
                }

                neuralNetwork.Backpropagate(inputs.ToArray(), targets, ErrorFunctionType.CrossEntropy, 0.05, 0.9);
            }
        }

        /// <summary>
        /// Calculates the similarity matrix of i and j (Qij)
        /// </summary>
        private static Dictionary<string, Dictionary<string, double>> CalculateSimilarityMatrix(
            Dictionary<string, Dictionary<string, double>> probabilityMatrix,
            Layer neuralNetwork
            )
        {
            var inputNodes = neuralNetwork.PreviousLayers[0].Nodes;
            var outputNodes = neuralNetwork.Nodes;

            var inverseSquareDistanceMatrix = new Dictionary<string, Dictionary<string, double>>();
            var i = 0;
            foreach (var (labelI, probabilitiesI) in probabilityMatrix)
            {
                var inverseSquareDistancesI = new Dictionary<string, double>();
                var j = 0;
                foreach (var (labelJ, _) in probabilitiesI)
                {
                    var squareDistance = 0d;
                    foreach (var outputNode in outputNodes)
                    {
                        var distance = outputNode.Weights[inputNodes[i]].Value - outputNode.Weights[inputNodes[j]].Value;
                        squareDistance += distance * distance;
                    }
                    inverseSquareDistancesI.Add(labelJ, 1d / (1d + squareDistance));
                    j++;
                }
                inverseSquareDistanceMatrix.Add(labelI, inverseSquareDistancesI);
                i++;
            }

            var similarityMatrix = new Dictionary<string, Dictionary<string, double>>();
            foreach(var (labelI, probabilitiesI) in probabilityMatrix)
            {
                var inverseSquareDistancesI = inverseSquareDistanceMatrix[labelI];
                var denominator = inverseSquareDistancesI.Sum(isd => isd.Value);

                var similaritiesI = new Dictionary<string, double>();
                foreach (var (labelJ, _) in probabilitiesI)
                {
                    var inverseSquareDistanceIJ = inverseSquareDistancesI[labelJ];
                    similaritiesI.Add(labelJ, inverseSquareDistanceIJ / (denominator - inverseSquareDistanceIJ));
                }
                similarityMatrix.Add(labelI, similaritiesI);
            }

            return similarityMatrix;
        }

        /// <summary>
        /// Calculates a matrix which organises probabilities proportional to the similarity of each vector pair.
        /// </summary>
        private static Dictionary<string, Dictionary<string, double>> CalculateProbabilityMatrix(
            IReadOnlyCollection<IEmbedding> embeddings,
            DistanceFunctionType distanceFunctionType,
            double perplexity)
        {
            var distanceMatrix = CalculateDistanceMatrix(embeddings, distanceFunctionType);
            var logPerplexity = Math.Log(perplexity);

            var probabilityMatrix = new Dictionary<string, Dictionary<string, double>>();
            foreach (var embedding in embeddings)
            {
                var probabilities = embeddings.ToDictionary(
                    otherIEmbedding => otherIEmbedding.Label, 
                    otherIEmbedding => 0d);
                CalculateProbabilities(
                    embedding.Label,
                    distanceMatrix[embedding.Label],
                    probabilities,
                    perplexity,
                    logPerplexity);
                probabilityMatrix.Add(embedding.Label, probabilities);
            }

            var sum = 0d;
            foreach (var labelI in embeddings.Select(w => w.Label))
            {
                var probabilitiesGivenI = probabilityMatrix[labelI];
                foreach (var labelJ in embeddings.Select(w => w.Label))
                {
                    sum += probabilitiesGivenI[labelJ] += probabilityMatrix[labelJ][labelI];
                }
            }

            var scale = 0.5 / sum;
            foreach (var labelI in embeddings.Select(w => w.Label))
            {
                var probabilitiesGivenI = probabilityMatrix[labelI];
                foreach (var labelJ in embeddings.Select(w => w.Label))
                {
                    probabilitiesGivenI[labelJ] = probabilityMatrix[labelJ][labelI] = Math.Max(probabilityMatrix[labelJ][labelI] * scale, MinPij);
                }
            }

            return probabilityMatrix;
        }

        /// <summary>
        /// Calculates a matrix which organises the distances between data points in the embeddings.
        /// </summary>
        private static Dictionary<string, Dictionary<string, double>> CalculateDistanceMatrix(
            IReadOnlyCollection<IEmbedding> embeddings,
            DistanceFunctionType distanceFunctionType)
        {
            var distanceFunction = DistanceFunctionResolver.ResolveDistanceFunction(distanceFunctionType);

            var matrix = new Dictionary<string, Dictionary<string, double>>();

            foreach (var embedding in embeddings)
            {
                var distances = embeddings.ToDictionary(
                    otherIEmbedding => otherIEmbedding.Label,
                    otherIEmbedding => distanceFunction.Invoke(embedding.Vector, otherIEmbedding.Vector));

                matrix.Add(embedding.Label, distances);
            }

            return matrix;
        }

        /// <summary>
        /// Uses binary search on the kernel bandwidth sigma to obtain the desired perplexity.
        /// Estimates beta and then calculates the Guassian Observed Perplexity.
        /// </summary>
        private static void CalculateProbabilities(
            string label,
            Dictionary<string, double> distances,
            Dictionary<string, double> probabilities,
            double perplexity,
            double logPerplexity)
        {
            var beta = EstimateInitialBeta(distances, perplexity);
            var diff = CalculateGuassianObservedPerplexity(label, distances, probabilities, -beta) - logPerplexity;
            var betaMin = 0d;
            var betaMax = double.PositiveInfinity;
            for (var iteration = 0;
                iteration < PerplexityMaxIterations && Math.Abs(diff) > PerplexityError;
                ++iteration)
            {
                if (diff > 0)
                {
                    betaMin = beta;
                    beta += double.IsPositiveInfinity(betaMax) ? beta : ((betaMax - beta) * 0.5);
                }
                else
                {
                    betaMax = beta;
                    beta -= (beta - betaMin) * 0.5;
                }

                diff = CalculateGuassianObservedPerplexity(label, distances, probabilities, -beta) - logPerplexity;
            }
        }

        private static double CalculateGuassianObservedPerplexity(
            string label,
            IReadOnlyDictionary<string, double> distances,
            IDictionary<string, double> probabilities,
            double mBeta
        )
        {
            var probabilitySum = 0d;
            foreach (var (w, distance) in distances.Where(wd => wd.Key != label))
            {
                probabilitySum += (probabilities[w] = Math.Exp(distance * mBeta));
            }

            if (probabilitySum <= 0)
            {
                return double.NegativeInfinity;
            }

            var scalingFactor = 1d / probabilitySum;
            var sum = 0d;
            foreach (var distance in distances)
            {
                sum += distance.Value * (probabilities[distance.Key] *= scalingFactor);
            }

            return Math.Log(probabilitySum) - (mBeta * sum);
        }

        /// <summary>
        /// Estimate beta from the distances in a row.
        /// **lacks mathematical argument**
        /// Average distance is often too large, so scale the average distance as 2 * N / perplexity.
        /// Estimate beta as 1 / (scaled average distance)
        /// </summary>
        private static double EstimateInitialBeta(
            IDictionary<string, double> distances,
            double perplexity)
        {
            var sum = distances.Values.Sum(distance => distance * distance);

            return (perplexity * (distances.Count - 1)) / (2 * sum);
        }
    }
}
