using System.Collections.Generic;
using GingerbreadAI.NLP.Word2Vec.Embeddings;
using GingerbreadAI.NLP.Word2Vec.Extensions;
using Xunit;

namespace GingerbreadAI.NLP.Word2Vec.Test.Extensions
{
    public class ArticleEmbeddingsCollectionExtensionsShould
    {
        [Fact]
        public void AssignAnyVectorWhenAssignVectorsFromWeightedWordEmbeddings()
        {
            var wordEmbeddings = new[]
            {
                new WordEmbedding("a", new [] {1d}),
                new WordEmbedding("b", new [] {2d}),
                new WordEmbedding("c", new [] {3d}),
                new WordEmbedding("d", new [] {4d}),
                new WordEmbedding("e", new [] {5d}),
            };
            var articleEmbeddings = new List<ArticleEmbedding>
            {
                new ArticleEmbedding("1", "a b c d e"),
                new ArticleEmbedding("2", "b"), // cannot be a as tf-idf(a) = 0
                new ArticleEmbedding("3", "a b c"),
                new ArticleEmbedding("4", "a e"),
                new ArticleEmbedding("5", "e d c b a")
            };

            articleEmbeddings.AssignVectorsFromWeightedWordEmbeddings(wordEmbeddings);

            foreach (var articleEmbedding in articleEmbeddings)
            {
                foreach (var v in articleEmbedding.Vector)
                {
                    Assert.NotEqual(0d, v);
                }
            }
        }

        [Fact]
        public void NotAssignAnyVectorWhenAssignVectorsFromWeightedWordEmbeddings()
        {
            var wordEmbeddings = new[]
            {
                new WordEmbedding("v", new [] {1d}),
                new WordEmbedding("w", new [] {2d}),
                new WordEmbedding("x", new [] {3d}),
                new WordEmbedding("y", new [] {4d}),
                new WordEmbedding("z", new [] {5d}),
            };
            var articleEmbeddings = new List<ArticleEmbedding>
            {
                new ArticleEmbedding("1", "a b c d e"),
                new ArticleEmbedding("2", "a"),
                new ArticleEmbedding("3", "a b c"),
                new ArticleEmbedding("4", "a e"),
                new ArticleEmbedding("5", "e d c b a")
            };

            articleEmbeddings.AssignVectorsFromWeightedWordEmbeddings(wordEmbeddings);

            foreach (var articleEmbedding in articleEmbeddings)
            {
                foreach (var v in articleEmbedding.Vector)
                {
                    Assert.Equal(0d, v);
                }
            }
        }
    }
}
