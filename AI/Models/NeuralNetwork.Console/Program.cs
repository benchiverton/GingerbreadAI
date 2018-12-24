namespace Network.Console
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using BackPropagation;
    using NegativeSampling;
    using NeuralNetwork;
    using NeuralNetwork.Data;
    using NeuralNetwork.Library;
    using NeuralNetwork.Library.Extensions;
    using Word2Vec.Ben;

    public class Program
    {
        public static void Main()
        {
            var word2Vec = new Word2Vec("input.txt", "wordDictionaryFile.dic", 10, 4);

            word2Vec.TrainModel();

            var outputGenerator = new OutputGenerator("output.csv");
            outputGenerator.WriteOutput(word2Vec.WordCollection, word2Vec.Network);

            //var group = new Layer("Input", 1, new Layer[0]);
            //var inner1 = new Layer("Inner1", 3, new[] { group });
            //var inner2 = new Layer("Inner2", 3, new[] { group });
            ////var inner2 = new Layer("Inner2", 25, new[] { inner1 });
            //var output = new Layer("Output", 1, new[] { inner1, inner2 });

            //var rand = new Random();
            //LayerInitialiser.Initialise(rand, output);

            //var nodeLayerLogic = new LayerCalculator
            //{
            //    OutputLayer = output
            //};
            //Console.WriteLine(output.ToString(true));

            //var inputs = new double[100];
            //var initialResults = new double[100];
            //var finalResults = new double[100];
            //for (var i = 0; i < inputs.Length; i++)
            //{
            //    inputs[i] = (double)i / inputs.Length;
            //}

            //// initial results
            //for (var i = 0; i < inputs.Length; i++)
            //{
            //    initialResults[i] = nodeLayerLogic.GetResults(new[] { inputs[i] })[0];
            //}

            //// perform backpropagation
            //var backpropagator = new BackPropagator(output, 1, LearningRateModifier, 0.9);
            //for (var i = 0; i < 1000000; i++)
            //{
            //    var trial = rand.NextDouble();
            //    backpropagator.BackPropagate(new[] { trial }, new[] { Calculation(trial) });
            //}

            //// final results
            //for (var i = 0; i < inputs.Length; i++)
            //{
            //    finalResults[i] = nodeLayerLogic.GetResults(new[] { inputs[i] })[0];
            //}

            //using (var file = new System.IO.StreamWriter($@"{Directory.GetCurrentDirectory()}/networkResults.csv", false))
            //{
            //    file.WriteLine(string.Join(",", inputs.ToArray()));
            //    file.WriteLine(string.Join(",", inputs.Select(Calculation)));
            //    file.WriteLine(string.Join(",", initialResults.ToArray()));
            //    file.WriteLine(string.Join(",", finalResults.ToArray()));
            //}
        }

        private static double LearningRateModifier(double rate)
            => rate * 0.99 < 0.1 ? 0.1 : rate * 0.99;

        private static double Calculation(double input)
            => 0.5 * Math.Sin(3 * Math.PI * input) + 0.5;
    }
}