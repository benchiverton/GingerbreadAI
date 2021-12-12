using System.Collections.Generic;
using System.Linq;
using GingerbreadAI.NLP.Word2Vec.Embeddings;

namespace GingerbreadAI.NLP.Word2Vec.Extensions;

public static class ArticleEmbeddingCollectionExtensions
{
    /// <summary>
    /// Assigns the vectors for article embeddings given the word embeddings.
    /// Weights the first 'numberOfWordsToUse' words in the article using TF-IDF.
    /// </summary>
    public static void AssignVectorsFromWeightedWordEmbeddings(this IEnumerable<ArticleEmbedding> articles, IEnumerable<WordEmbedding> wordEmbeddings, int numberOfWordsToUse = 50)
    {
        var wordEmbeddingsDictionary = wordEmbeddings.ToDictionary();
        var vectorDimension = wordEmbeddingsDictionary.First().Value.Length;
        var articlesList = articles.ToList();

        foreach (var article in articlesList)
        {
            var wordCount = 0;
            var embedding = new double[vectorDimension];
            foreach (var word in article.Contents.GetWords().Take(numberOfWordsToUse))
            {
                if (wordEmbeddingsDictionary.TryGetValue(word, out var wordEmbedding))
                {
                    wordCount++;
                    var tfidf = article.Contents.CalculateTFIDF(
                        word,
                        articlesList.Select(a => a.Contents).ToList());
                    embedding = embedding.Zip(wordEmbedding, (x, y) => x + (tfidf * y)).ToArray();
                }
            }

            article.Vector = embedding;
        }
    }
}
