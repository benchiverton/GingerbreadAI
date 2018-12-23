using NegativeSampling;
using NeuralNetwork;
using NeuralNetwork.Data;
using System;
using Xunit;

namespace BackPropagation.Test
{
    public class BackPropagatorShould
    {
        [Fact]
        public void TrainNetworksSortofWell()
        {
            var input = new Layer("input", 5, new Layer[0]);
            var h1 = new Layer("hidden", 10, new Layer[] { input });
            var output = new Layer("output", 5, new Layer[] { h1 });

            LayerInitialiser.Initialise(new Random(), output);

            var og = new OutputCalculator(output);
            var ns = new NegativeSampler(output, 0.25);
            
            for (int i = 0; i < 2000; i++)
            {
                ns.NegativeSample(0, 0, false);
                ns.NegativeSample(1, 1, false);
                ns.NegativeSample(2, 2, true);
                ns.NegativeSample(3, 3, false);
                ns.NegativeSample(4, 4, false);
            }

            Assert.True(og.GetResult(0, 0) < 0.05);
            Assert.True(og.GetResult(1, 1) < 0.05);
            Assert.True(og.GetResult(2, 2) > 0.95);
            Assert.True(og.GetResult(3, 3) < 0.05);
            Assert.True(og.GetResult(4, 4) < 0.05);
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
            var ns = new NegativeSampler(output, 0.25);

            for (int i = 0; i < 2000; i++)
            {
                ns.NegativeSample(0, 0, false);
                ns.NegativeSample(1, 1, false);
                ns.NegativeSample(2, 2, true);
                ns.NegativeSample(3, 3, false);
                ns.NegativeSample(4, 4, false);
            }

            Assert.True(og.GetResult(0, 0) < 0.05);
            Assert.True(og.GetResult(1, 1) < 0.05);
            Assert.True(og.GetResult(2, 2) > 0.95);
            Assert.True(og.GetResult(3, 3) < 0.05);
            Assert.True(og.GetResult(4, 4) < 0.05);
        }
    }
}