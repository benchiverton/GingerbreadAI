using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using NeuralNetwork;
using NeuralNetwork.Data;

namespace Word2Vec.Ben
{
    public class OutputGenerator
    {
        private readonly string _outputFile;

        public OutputGenerator(string outputFile)
        {
            _outputFile = outputFile;
        }

        public void WriteOutput(WordCollection wordCollection, Layer network)
        {
            using (var fs = new FileStream(_outputFile, FileMode.Create, FileAccess.Write))
            using (var writer = new StreamWriter(fs, Encoding.UTF8))
            {
                string line = "##";
                var wordArray = wordCollection.GetWords().ToArray();
                for (var i = 0; i < wordCollection.GetNumberOfUniqueWords(); i++)
                {
                    line += $",{wordArray[i]}";
                }
                writer.WriteLine(line);

                for (var i = 0; i < wordCollection.GetNumberOfUniqueWords(); i++)
                {
                    line = wordArray[i];

                    var probabilitiesForSomeWord = GetProbabilities(network, wordCollection, i);
                    foreach (var r in probabilitiesForSomeWord)
                    {
                        line += $",{r.Item2}";
                    }

                    writer.WriteLine(line);
                }
            }
        }

        private List<(string, double)> GetProbabilities(Layer network, WordCollection wordCollection, int index)
        {
            var inputs = new double[wordCollection.GetNumberOfUniqueWords()];
            for (var i = 0; i < inputs.Length; i++)
            {
                inputs[i] = int.MinValue;
            }
            inputs[index] = int.MaxValue;

            var layerCalculator = new LayerCalculator(network);
            var outputs = new List<(string, double)>();
            var results = layerCalculator.GetResults(inputs);
            var words = wordCollection.GetWords().ToArray();
            for (var i = 0; i < words.Length; i++)
            {
                outputs.Add((words[i], results[i]));
            }

            return outputs;
        }
    }
}
