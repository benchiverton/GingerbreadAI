namespace Word2Vec.Ben
{
    using NeuralNetwork;
    using NeuralNetwork.Data;
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Text;

    public class FileHandler
    {
        private readonly string _trainFile;
        private readonly string _outputFile;
        public long FileSize { get; }

        public FileHandler(string trainFile, string outputFile)
        {
            _trainFile = trainFile;
            _outputFile = outputFile;

            FileSize = new FileInfo(_trainFile).Length;

            if (string.IsNullOrEmpty(_outputFile))
                throw new Exception("Output file not defined.");
        }

        public void GetWordDictionaryFromFile(WordCollection wordCollection,
            int maxCodeLength)
        {
            if (!File.Exists(_trainFile))
                throw new InvalidOperationException($"Unable to find {_trainFile}");

            using (var fileStream = new FileStream(_trainFile, FileMode.Open, FileAccess.Read))
            using (var reader = new StreamReader(fileStream, Encoding.UTF8))
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

        public void WriteOutput(WordCollection wordCollection, Layer network)
        {
            var words = wordCollection.GetWords().ToArray();

            var result = new StringBuilder();
            result.Append(',').AppendJoin(',', words).AppendLine();
            for (int i = 0; i < words.Length; i++)
            {
                result.Append(words[i]);
                for (int j = 0; j < words.Length; j++)
                {
                    result.Append($",{network.GetResult(i, j)}");
                }
                result.AppendLine();
            }

            using (var fs = new FileStream(_outputFile, FileMode.Create, FileAccess.Write))
            using (var writer = new StreamWriter(fs, Encoding.UTF8))
            {
                writer.WriteLine($"Unique Words: {wordCollection.GetNumberOfUniqueWords()}");
                writer.WriteLine($"Network used:");
                writer.WriteLine(network.ToString());
                writer.WriteLine();
                writer.WriteLine(result.ToString());
            }
        }

        public StreamReader GetReader()
        {
            return File.OpenText(_trainFile);
        }
    }
}