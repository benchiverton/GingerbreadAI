using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace GingerbreadAI.NLP.Word2Vec.Extensions
{
    public static class WordEmbeddingCollectionExtensions
    {
        /// <summary>
        /// Writes word embedding collection to the word embeddings file.
        /// </summary>
        public static void WriteWordEmbeddingsToFile(this IEnumerable<WordEmbedding> wordEmbeddings, string wordEmbeddingsFile)
        {
            using (var fs = new FileStream(wordEmbeddingsFile, FileMode.OpenOrCreate, FileAccess.Write))
            {
                fs.Seek(0, SeekOrigin.End);
                using (var writer = new StreamWriter(fs, Encoding.UTF8))
                {
                    foreach (var wordEmbedding in wordEmbeddings)
                    {
                        writer.WriteLine($"{wordEmbedding.Word},{string.Join(',', wordEmbedding.Vector)}");
                    }
                }
            }
        }

        /// <summary>
        /// Populates word embedding collection from the word embeddings file.
        /// </summary>
        public static void PopulateWordEmbeddingsFromFile(this List<WordEmbedding> wordEmbeddings, string wordEmbeddingsFile)
        {
            using (var streamReader = File.OpenText(wordEmbeddingsFile))
            {
                while (!streamReader.EndOfStream)
                {
                    var lineElements = streamReader.ReadLine()?.Split(',') ?? throw new Exception("Could not read a word embedding from the file provided.");
                    wordEmbeddings.Add(new WordEmbedding(
                        lineElements[0],
                        lineElements.Skip(1).Select(double.Parse).ToArray()
                    ));
                }
            }
        }
    }
}
