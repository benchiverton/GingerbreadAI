using System;
using NeuralNetwork.Data;
using NeuralNetwork.Library;

namespace Main
{
    public class Program
    {
        public static void Main()
        {

            //var group = new NodeGroup("Input", 20);
            //var inner1 = new NodeGroup("Inner1", 20, new[] { group });
            //var inner2 = new NodeGroup("Inner2", 20, new[] { group });
            //var output = new NodeGroup("Output", 20, new[] { inner1, inner2 });

            //Initialiser.Initialise(new Random(), output);

            //var inputs = new double[20];
            //for (var i = 0; i < inputs.Length; i++)
            //{
            //    inputs[i] = 1;
            //}
            //Console.WriteLine(output.ToString(true));
            //NodeGroupCalculations.GetResult(output, inputs);
            //var results = output.Outputs;
            //Console.WriteLine($"Results: {string.Join(", ", results)}");

            var h1 = 0.978751677288986;
            var h2 = 0.99742672684619;
            var h3 = 0.99951940263283;

            var x = new[] { 0.02 * 41 + 0.13 * 43 };
            foreach (var y in x)
            {
                Console.WriteLine(NodeCalculations.LogisticFunction(y) + "\n");
            }

            Console.ReadLine();
        }
    }
}