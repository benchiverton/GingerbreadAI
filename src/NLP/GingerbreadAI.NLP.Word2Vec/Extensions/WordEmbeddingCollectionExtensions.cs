using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using GingerbreadAI.NLP.Word2Vec.Embeddings;

namespace GingerbreadAI.NLP.Word2Vec.Extensions;

public static class WordEmbeddingCollectionExtensions
{
    /// <summary>
    /// Writes word embeddings to the stream.
    /// </summary>
    public static void WriteEmbeddingToStream(this IEnumerable<WordEmbedding> wordEmbeddings, StreamWriter streamWriter)
    {
        foreach (var wordEmbedding in wordEmbeddings)
        {
            var content = string.Join(',', wordEmbedding.Vector.Select(p => p.ToString(CultureInfo.CreateSpecificCulture("en-GB"))).ToArray());
            streamWriter.WriteLine($"{wordEmbedding.Label},{content}");
        }
    }

    /// <summary>
    /// Populates embeddings from stream.
    /// </summary>
    public static void PopulateWordEmbeddingsFromStream(this List<WordEmbedding> wordEmbeddings, StreamReader streamReader)
    {
        while (!streamReader.EndOfStream)
        {
            var lineElements = streamReader.ReadLine()?.Split(',') ?? throw new Exception("Could not read an embedding from the file provided.");
            wordEmbeddings.Add(new WordEmbedding(
                lineElements[0],
                lineElements.Skip(1).Select((s, i) => double.Parse(s, CultureInfo.CreateSpecificCulture("en-GB"))).ToArray()
            ));
        }
    }
}
