using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using GingerbreadAI.NLP.Word2Vec.Embeddings;

namespace GingerbreadAI.NLP.Word2Vec.Extensions
{
    public static class WordEmbeddingCollectionExtensions
    {
        /// <summary>
        /// Writes word embeddings to the embeddings file.
        /// </summary>
        public static void WriteWordEmbeddingsToFile(this IEnumerable<WordEmbedding> wordEmbeddings, string embeddingsFile)
        {
            using (var fs = new FileStream(embeddingsFile, FileMode.OpenOrCreate, FileAccess.Write))
            {
                fs.Seek(0, SeekOrigin.End);
                using (var writer = new StreamWriter(fs, Encoding.UTF8))
                {
                    foreach (var embedding in wordEmbeddings)
                    {
                        writer.WriteLine($"{embedding.Label},{string.Join(',', embedding.Vector)}");
                    }
                }
            }
        }

        /// <summary>
        /// Populates embeddings from the embeddings file.
        /// </summary>
        public static void PopulateWordEmbeddingsFromFile(this List<WordEmbedding> wordEmbeddings, string embeddingsFile)
        {
            using (var streamReader = File.OpenText(embeddingsFile))
            {
                while (!streamReader.EndOfStream)
                {
                    var lineElements = streamReader.ReadLine()?.Split(',') ?? throw new Exception("Could not read an embedding from the file provided.");
                    wordEmbeddings.Add(new WordEmbedding(
                        lineElements[0],
                        lineElements.Skip(1).Select(double.Parse).ToArray()
                    ));
                }
            }
        }
    }
}
