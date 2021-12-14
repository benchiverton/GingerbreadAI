using System;

namespace GingerbreadAI.Model.NeuralNetwork.InitialisationFunctions;

public static class InitialisationFunctionResolver
{
    public static Func<Random, int, int, double> ResolveInitialisationFunctions(InitialisationFunctionType initialisationFunctionType) =>
        initialisationFunctionType switch
        {
            InitialisationFunctionType.None => (rand, feedingNodeCount, layerNodeCount)
                => 0d,
            InitialisationFunctionType.Random => (rand, feedingNodeCount, layerNodeCount)
                => (rand.NextDouble() * 2) - 1,
            InitialisationFunctionType.RandomWeighted => (rand, feedingNodeCount, layerNodeCount)
                => (rand.NextDouble() - 0.5) / layerNodeCount,
            InitialisationFunctionType.RandomPositive => (rand, feedingNodeCount, layerNodeCount)
                => rand.NextDouble(),
            InitialisationFunctionType.HeUniform => (rand, feedingNodeCount, layerNodeCount)
                => Math.Sqrt(6d / (feedingNodeCount)) * ((rand.NextDouble() * 2) - 1),
            InitialisationFunctionType.GlorotUniform => (rand, feedingNodeCount, layerNodeCount)
                => Math.Sqrt(6d / ((double)feedingNodeCount + layerNodeCount)) * ((rand.NextDouble() * 2) - 1),
            InitialisationFunctionType.HeEtAl => (rand, feedingNodeCount, layerNodeCount)
                => (2d * rand.NextDouble() / Math.Sqrt(feedingNodeCount)) - (1d / Math.Sqrt(feedingNodeCount)),
            InitialisationFunctionType.RandomGuassian => (rand, feedingNodeCount, layerNodeCount)
                => RandomGuassian(rand),
            _ => throw new ArgumentOutOfRangeException(
                nameof(initialisationFunctionType),
                initialisationFunctionType,
                "This initialisation function type is not yet supported. Please use a different initialisation function type.")
        };

    /// <summary>
    /// Returns a random Guassian integer using the Box-Muller transformation.
    /// Mean = 0; Variance = 1
    /// </summary>
    private static double RandomGuassian(Random rand)
    {
        var u1 = 1.0 - rand.NextDouble(); //uniform(0,1] random doubles
        var u2 = 1.0 - rand.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) *
                            Math.Sin(2.0 * Math.PI * u2);
    }
}
