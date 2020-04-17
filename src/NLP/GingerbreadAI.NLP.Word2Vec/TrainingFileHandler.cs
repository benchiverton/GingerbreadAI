using System;
using System.IO;
using System.Text;

namespace GingerbreadAI.NLP.Word2Vec
{
    public class TrainingFileHandler
    {
        private readonly string _trainingFile;

        public TrainingFileHandler(string trainingFile)
        {
            _trainingFile = trainingFile;

            TrainingFileSize = new FileInfo(_trainingFile).Length;
        }

        public long TrainingFileSize { get; }

        public WordCollection GetWordDictionaryFromFile(int maxCodeLength)
        {
            var wordCollection = new WordCollection();

            if (!File.Exists(_trainingFile))
            {
                throw new InvalidOperationException($"Unable to find {_trainingFile}");
            }

            using (var fileStream = new FileStream(_trainingFile, FileMode.OpenOrCreate, FileAccess.Read))
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
            return File.OpenText(_trainingFile);
        }
    }
}