using System;
using System.IO;
using System.Text;

namespace Word2Vec
{
    public class FileHandler
    {
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
    }
}