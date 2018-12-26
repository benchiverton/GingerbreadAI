using NeuralNetwork;
using NeuralNetwork.Data;
using System;
using Xunit;

namespace BackPropagation.Test
{
    public class BackPropagatorShould
    {
        [Fact]
        public void TrainBasicNetworksSortofWell()
        {
            var input = new Layer("input", 5, new Layer[0]);
            var h1 = new Layer("hidden", 10, new Layer[] { input });
            var output = new Layer("output", 5, new Layer[] { h1 });

            LayerInitialiser.Initialise(new Random(), output);

            var og = new OutputCalculator(output);
            var bp = new BackPropagator(output, 0.25);

            var inputs = new double[] { 0.1, 0.2, 0.3, 0.4, 0.5 };
            var targetOutputs = new double?[] { 1, 0, 1, 0, 1 };
            for(int i=0; i<1000; i++)
            {
                bp.BackPropagate(inputs, targetOutputs);
            }

            var outputResults = og.GetResults(inputs);

            Assert.True(outputResults[0] > targetOutputs[0] - 0.05);
            Assert.True(outputResults[2] > targetOutputs[2] - 0.05);
            Assert.True(outputResults[4] > targetOutputs[4] - 0.05);

            Assert.True(outputResults[1] < targetOutputs[1] + 0.05);
            Assert.True(outputResults[3] < targetOutputs[3] + 0.05);
        }

        [Fact]
        public void TrainComplexNetworksSortofWell()
        {
            var input = new Layer("input", 5, new Layer[0]);
            var h1 = new Layer("hidden1", 10, new Layer[] { input });
            var h2 = new Layer("hidden2", 10, new Layer[] { input });
            var h3 = new Layer("hidden3", 10, new Layer[] { h1, h2 });
            var output = new Layer("output", 5, new Layer[] { h3 });

            LayerInitialiser.Initialise(new Random(), output);

            var og = new OutputCalculator(output);
            var bp = new BackPropagator(output, 0.25);

            var inputs = new double[] { 0.1, 0.2, 0.3, 0.4, 0.5 };
            var targetOutputs = new double?[] { 1, 0, 1, 0, 1 };
            for (int i = 0; i < 1000; i++)
            {
                bp.BackPropagate(inputs, targetOutputs);
            }

            var outputResults = og.GetResults(inputs);

            Assert.True(outputResults[0] > targetOutputs[0] - 0.05);
            Assert.True(outputResults[2] > targetOutputs[2] - 0.05);
            Assert.True(outputResults[4] > targetOutputs[4] - 0.05);

            Assert.True(outputResults[1] < targetOutputs[1] + 0.05);
            Assert.True(outputResults[3] < targetOutputs[3] + 0.05);
        }
    }
}