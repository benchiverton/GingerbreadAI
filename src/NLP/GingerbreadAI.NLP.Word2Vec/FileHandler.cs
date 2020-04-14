using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using GingerbreadAI.Model.NeuralNetwork.Models;

namespace GingerbreadAI.NLP.Word2Vec
{
    public class FileHandler
    {
        private readonly string _trainFile;
        private readonly string _outputFile;

        public FileHandler(string trainFile, string outputFile)
        {
            _trainFile = trainFile;
            _outputFile = outputFile;

            FileSize = new FileInfo(_trainFile).Length;

            if (string.IsNullOrEmpty(_outputFile))
            {
                throw new Exception("Output file not defined.");
            }
        }

        public long FileSize { get; }

        public WordCollection GetWordDictionaryFromFile(int maxCodeLength)
        {
            var wordCollection = new WordCollection();

            if (!File.Exists(_trainFile))
            {
                throw new InvalidOperationException($"Unable to find {_trainFile}");
            }

            using (var fileStream = new FileStream(_trainFile, FileMode.OpenOrCreate, FileAccess.Read))
            {
                using (var reader = new StreamReader(fileStream, Encoding.UTF8))
                {
                    string line;
                    while ((line = reader.ReadLine()) != null)
                    {
                        wordCollection.AddWords(line, maxCodeLength);

                        if (reader.EndOfStream)
                        {
                            break;
                        }
                    }
                }
            }

            return wordCollection;
        }

        public StreamReader GetTrainingFileReader()
        {
            return File.OpenText(_trainFile);
        }

        internal FileStream GetOutputFileStream()
        {
            return new FileStream(_outputFile, FileMode.OpenOrCreate, FileAccess.Write);
        }
    }
}