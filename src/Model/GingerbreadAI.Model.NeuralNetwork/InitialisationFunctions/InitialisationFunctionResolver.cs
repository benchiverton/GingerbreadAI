using System;

namespace GingerbreadAI.Model.NeuralNetwork.InitialisationFunctions
{
    public static class InitialisationFunctionResolver
    {
        public static Func<Random, int, int, double> ResolveInitialisationFunctions(InitialisationFunctionType initialisationFunctionType)
        {
            return initialisationFunctionType switch
            {
                InitialisationFunctionType.None => (rand, feedingNodeCount, layerNodeCount) 
                    => 0d,
                InitialisationFunctionType.Random => (rand, feedingNodeCount, layerNodeCount)
                    => rand.NextDouble() * 2 - 1,
                InitialisationFunctionType.RandomPositive => (rand, feedingNodeCount, layerNodeCount)
                    => rand.NextDouble(),
                InitialisationFunctionType.HeUniform => (rand, feedingNodeCount, layerNodeCount)
                    => Math.Sqrt(6d / (feedingNodeCount)) * (rand.NextDouble() * 2 - 1),
                InitialisationFunctionType.GlorotUniform => (rand, feedingNodeCount, layerNodeCount)
                    => Math.Sqrt(6d / ((double)feedingNodeCount + layerNodeCount)) * (rand.NextDouble() * 2 - 1),
                InitialisationFunctionType.HeEtAl => (rand, feedingNodeCount, layerNodeCount)
                    => (2d * rand.NextDouble() / Math.Sqrt(feedingNodeCount)) - 1d / Math.Sqrt(feedingNodeCount),
                _ => throw new ArgumentOutOfRangeException(
                        nameof(initialisationFunctionType),
                        initialisationFunctionType,
                        "This initialisation function type is not yet supported. Please use a different initialisation function type.")
            };
        }
    }
}
