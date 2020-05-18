using System;
using System.IO;
using System.Text;

namespace GingerbreadAI.NLP.Word2Vec
{
    public class FileHandler
    {
        private readonly string _file;

        public FileHandler(string file)
        {
            _file = file;

            FileSize = new FileInfo(_file).Length;
        }

        public long FileSize { get; }

        public WordCollection GetWordDictionaryFromFile(int maxCodeLength)
        {
            var wordCollection = new WordCollection();

            if (!File.Exists(_file))
            {
                throw new InvalidOperationException($"Unable to find {_file}");
            }

            using (var fileStream = new FileStream(_file, FileMode.OpenOrCreate, FileAccess.Read))
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

        public StreamReader GetTrainingFileReader() => File.OpenText(_file);
    }
}
