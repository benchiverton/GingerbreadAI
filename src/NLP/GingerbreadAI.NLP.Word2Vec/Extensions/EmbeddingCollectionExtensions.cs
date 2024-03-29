using System;
using System.Collections.Generic;
using System.Linq;
using GingerbreadAI.NLP.Word2Vec.Embeddings;
using GingerbreadAI.NLP.Word2Vec.SimilarityFunctions;

namespace GingerbreadAI.NLP.Word2Vec.Extensions;

public static class EmbeddingCollectionExtensions
{
    /// <summary>
    /// Normalises the length of each embedding in the collection.
    /// </summary>
    public static void NormaliseEmbeddings(this IEnumerable<IEmbedding> embeddings)
    {
        foreach (var embedding in embeddings)
        {
            var length = 0d;
            foreach (var v in embedding.Vector)
            {
                length += v * v;
            }
            length = Math.Sqrt(length);
            for (var i = 0; i < embedding.Vector.Length; i++)
            {
                embedding.Vector[i] /= length;
            }
        }
    }

    /// <summary>
    /// Returns the top n most similar embeddings to the one given.
    /// </summary>
    public static IEnumerable<(IEmbedding embedding, double similarity)> GetMostSimilarEmbeddings(
        this IEnumerable<IEmbedding> embeddings,
        IEmbedding embedding,
        int n = 10,
        SimilarityFunctionType similarityFunctionType = SimilarityFunctionType.Cosine)
    {
        var similarityFunction = SimilarityFunctionResolver.ResolveSimilarityFunction(similarityFunctionType);

        var embeddingsArray = embeddings.ToArray();

        return embeddingsArray.Where(we => we.Label != embedding.Label)
            .Select(otherEmbedding => (otherEmbedding, similarityFunction.Invoke(embedding.Vector.ToArray(), otherEmbedding.Vector.ToArray())))
            .OrderByDescending(owcs => owcs.Item2)
            .Take(n);
    }

    /// <summary>
    /// Converts a list of embeddings to a dictionary.
    /// </summary>
    public static Dictionary<string, double[]> ToDictionary(this IEnumerable<IEmbedding> embeddings) => embeddings.ToDictionary(embedding => embedding.Label, embedding => embedding.Vector);
}
