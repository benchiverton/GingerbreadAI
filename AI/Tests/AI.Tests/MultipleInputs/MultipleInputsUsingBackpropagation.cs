using NeuralNetwork;
using NeuralNetwork.Models;
using System;
using Xunit.Abstractions;

namespace AI.Tests.MultipleInputs
{
    public class MultipleInputsUsingBackpropagation
    {
        private const string ResultsDirectory = nameof(MultipleInputsUsingBackpropagation);
        private readonly ITestOutputHelper _testOutputHelper;

        public MultipleInputsUsingBackpropagation(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        [RunnableInDebugOnly]
        public void PredictResultsFromMultipleInputs()
        {
            var input1 = new Layer("RandomWalk1", 5, new Layer[0]);
            var input2 = new Layer("RandomWalk2", 5, new Layer[0]);
            var inner = new Layer("Inner", 20, new[] { input1, input2 });
            var outputLayer = new Layer("Output", 1, new[] { inner });
            LayerInitialiser.Initialise(new Random(), outputLayer);
            _testOutputHelper.WriteLine(outputLayer.ToString(true));
        }

        private class RandomWalker
        {
            public RandomWalker()
            {
                Position = 0;
            }

            public double Position { get; private set; }

            public void Walk(double probabilityOfIncreasing, Random random)
            {
                if (random.NextDouble() < probabilityOfIncreasing)
                {
                    Position++;
                }
                else
                {
                    Position--;
                }
            }
        }
    }
}
