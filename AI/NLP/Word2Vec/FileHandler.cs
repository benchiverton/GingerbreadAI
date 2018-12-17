using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace Word2Vec
{
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

        public void WriteOutput(WordCollection wordCollection, int numberOfDimensions, float[,] hiddenLayerWeights)
        {
            using (var fs = new FileStream(_outputFile, FileMode.Create, FileAccess.Write))
            using (var writer = new StreamWriter(fs, Encoding.UTF8))
            {
                writer.WriteLine(wordCollection.GetNumberOfUniqueWords());
                writer.WriteLine(numberOfDimensions);

                var keys = wordCollection.GetWords().ToArray();
                for (var a = 0; a < wordCollection.GetNumberOfUniqueWords(); a++)
                {
                    var bytes = new List<byte>();
                    for (var dimensionIndex = 0; dimensionIndex < numberOfDimensions; dimensionIndex++)
                        bytes.AddRange(BitConverter.GetBytes(hiddenLayerWeights[a, dimensionIndex]));
                    writer.WriteLine($"{keys[a]}\t{Convert.ToBase64String(bytes.ToArray())}");
                }
            }
        }

        public StreamReader GetReader()
        {
            return File.OpenText(_trainFile);
        }
    }
}