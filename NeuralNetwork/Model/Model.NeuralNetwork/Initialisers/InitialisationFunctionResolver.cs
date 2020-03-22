using System;

namespace Model.NeuralNetwork.Initialisers
{
    public static class InitialisationFunctionResolver
    {
        public static Func<Random, int, int, double> ResolveInitialisationFunctions(InitialisationFunctionType initialisationFunctionType)
        {
            Func<Random, int, int, double> initialisationFunction;
            switch (initialisationFunctionType)
            {
                case InitialisationFunctionType.None:
                    initialisationFunction = (rand, feedingNodeCount, outputNodeCount) => 0d;
                    break;
                case InitialisationFunctionType.Random:
                    initialisationFunction = (rand, feedingNodeCount, outputNodeCount) => rand.NextDouble() * 2 - 1;
                    break;
                case InitialisationFunctionType.RandomPositive:
                    initialisationFunction = (rand, feedingNodeCount, outputNodeCount) => rand.NextDouble();
                    break;
                case InitialisationFunctionType.HeUniform:
                    initialisationFunction = (rand, feedingNodeCount, outputNodeCount) => Math.Sqrt(6d / (feedingNodeCount)) * (rand.NextDouble() * 2 - 1);
                    break;
                case InitialisationFunctionType.GlorotUniform:
                    initialisationFunction = (rand, feedingNodeCount, outputNodeCount) => Math.Sqrt(6d / ((double)feedingNodeCount + outputNodeCount)) * (rand.NextDouble() * 2 - 1);
                    break;
                case InitialisationFunctionType.HeEtAl:
                    initialisationFunction = (rand, feedingNodeCount, outputNodeCount) => (2d * rand.NextDouble() / Math.Sqrt(feedingNodeCount)) - 1d / Math.Sqrt(feedingNodeCount);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(
                        nameof(initialisationFunctionType),
                        initialisationFunctionType,
                        "This initialisation function type is not yet supported. Please use a different initialisation function type.");
            }

            return initialisationFunction;
        }
    }
}
