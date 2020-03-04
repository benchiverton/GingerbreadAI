using System;

namespace Model.NeuralNetwork.Initialisers
{
    public static class InitialisationFunctionResolver
    {
        public static Func<Random, int, double> ResolveInitialisationFunctions(InitialisationFunctionType initialisationFunctionType)
        {
            Func<Random, int, double> initialisationFunction;
            switch (initialisationFunctionType)
            {
                case InitialisationFunctionType.None:
                    initialisationFunction = (rand, feedingNodeCount) => 0d;
                    break;
                case InitialisationFunctionType.Random:
                    initialisationFunction = (rand, feedingNodeCount) => rand.NextDouble() * 2 - 1;
                    break;
                case InitialisationFunctionType.RandomPositive:
                    initialisationFunction = (rand, feedingNodeCount) => rand.NextDouble();
                    break;
                case InitialisationFunctionType.Uniform:
                    initialisationFunction = (rand, feedingNodeCount) => 1d / Math.Sqrt(feedingNodeCount);
                    break;
                case InitialisationFunctionType.HeEtAl:
                    initialisationFunction = (rand, feedingNodeCount) => (2d * rand.NextDouble() / Math.Sqrt(feedingNodeCount)) - 1d / Math.Sqrt(feedingNodeCount);
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
