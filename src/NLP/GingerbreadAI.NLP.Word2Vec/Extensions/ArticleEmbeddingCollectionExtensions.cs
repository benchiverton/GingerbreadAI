using System.Collections.Generic;
using System.Linq;
using GingerbreadAI.NLP.Word2Vec.Embeddings;

namespace GingerbreadAI.NLP.Word2Vec.Extensions
{
    public static class ArticleEmbeddingCollectionExtensions
    {
        /// <summary>
        /// Assigns the vectors for article embeddings given the word embeddings.
        /// Weights each word in the article using TF-IDF.
        /// </summary>
        public static void AssignWeightedVectorsFromWordEmbeddings(this IEnumerable<ArticleEmbedding> articles, IEnumerable<WordEmbedding> wordEmbeddings)
        {
            var wordEmbeddingsDictionary = wordEmbeddings.ToDictionary();
            var vectorDimension = wordEmbeddingsDictionary.First().Value.Length;
            var articlesList = articles.ToList();

            foreach (var article in articlesList)
            {
                var wordCount = 0;
                var embedding = new double[vectorDimension];
                foreach (var word in article.Contents.GetWords().Take(10))
                {
                    if (wordEmbeddingsDictionary.TryGetValue(word, out var wordEmbedding))
                    {
                        wordCount++;
                        var tfidf = article.Contents.CalculateTFIDF(
                            word,
                            articlesList.Select(a => a.Contents).ToList());
                        embedding = embedding.Zip(wordEmbedding, (x, y) => x + tfidf * y).ToArray();
                    }
                }

                embedding = embedding.Select(x => x / wordCount).ToArray();
                article.Vector = embedding;
            }
        }
    }
}
