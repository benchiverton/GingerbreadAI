using NeuralNetwork;
using NeuralNetwork.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace Word2Vec.Ben
{
    public class FileHandler
    {
        public static void WriteOutput(string outputFile, WordCollection wordCollection, Layer network)
        {
            Console.WriteLine($"writing output to {outputFile}...");
            using (var fs = new FileStream(outputFile, FileMode.Create, FileAccess.Write))
            using (var writer = new StreamWriter(fs, Encoding.UTF8))
            {
                var str = new StringBuilder("##");
                var words = wordCollection.GetWords().ToArray();
                for (var i = 0; i < words.Length; i++)
                {
                    str.Append($",{words[i]}");
                }
                writer.WriteLine(str.ToString());

                for (var i = 0; i < words.Length; i++)
                {
                    str = new StringBuilder(words[i]);

                    var probabilities = GetProbabilities(network, wordCollection, i);
                    foreach (var r in probabilities)
                    {
                        str.Append($",{r}");
                    }

                    writer.WriteLine(str.ToString());
                }
            }
        }

        public static void SaveWordDictionary(string vocabFileName, WordCollection wordCollection)
        {
            using (var stream = new FileStream(vocabFileName, FileMode.OpenOrCreate))
            using (var streamWriter = new StreamWriter(stream, Encoding.UTF8))
            {
                foreach (var word in wordCollection.GetWords())
                    streamWriter.WriteLine($"{word}\t{wordCollection.GetOccuranceOfWord(word)}");
            }
        }

        public static void GetWordDictionaryFromFile(string trainFileName, WordCollection wordCollection,
            int maxCodeLength)  
        {
            if (!File.Exists(trainFileName))
                throw new InvalidOperationException("ERROR: training data file not found!\n");

            using (var fs = new FileStream(trainFileName, FileMode.Open, FileAccess.Read))
            using (var reader = new StreamReader(fs, Encoding.UTF8))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    wordCollection.AddWords(line, maxCodeLength);

                    if (reader.EndOfStream)
                        break;
                }
            }
        }

        private static double[] GetProbabilities(Layer network, WordCollection wordCollection, int index)
        {
            var outputCalculator = new OutputCalculator(network);
            var words = wordCollection.GetWords().ToArray();
            var outputs = new double[words.Length];

            for (var i = 0; i < words.Length; i++)
            {
                outputs[i] = outputCalculator.GetResult(index, i);
            }

            return outputs;
        }
    }
}