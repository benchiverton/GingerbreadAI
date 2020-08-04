using System;
using System.Linq;
using GingerbreadAI.NLP.Word2Vec.Embeddings;
using GingerbreadAI.NLP.Word2Vec.Extensions;
using Xunit;

namespace GingerbreadAI.NLP.Word2Vec.Test.Extensions
{
    public class EmbeddingCollectionExtensionsShould
    {
        [Fact]
        public void CorrectlyNormaliseEmbeddings()
        {
            var wordEmbeddings = new[]
            {
                new WordEmbedding("a", new [] {1d, 2d, 3d, 4d}),
                new WordEmbedding("b", new [] {-1d, -1d}),
                new WordEmbedding("c", new [] {-0.1, 0.2d}),
                new WordEmbedding("d", new [] {50d, 100d}),
            };

            wordEmbeddings.NormaliseEmbeddings();

            foreach (var wordEmbedding in wordEmbeddings)
            {
                var magnitude = 0d;
                foreach (var v in wordEmbedding.Vector)
                {
                    magnitude += v * v;
                }
                Assert.Equal(1d, Math.Round(magnitude, 8));
            }
        }

        [Fact]
        public void CorrectlyGetMostSimilarWords()
        {
            var wordEmbeddings = new[]
            {
                new WordEmbedding("target", new [] {1d, 1d}),
                new WordEmbedding("far", new [] {-1d, -1d}),
                new WordEmbedding("close-ish", new [] {-1d, 1d}),
                new WordEmbedding("same", new [] {1d, 1d}),
            };

            var orderedWords = wordEmbeddings.GetMostSimilarEmbeddings(new WordEmbedding("target", new[] { 1d, 1d }), 3).ToArray();
            Assert.Equal(1d, orderedWords[0].similarity, 8);
            Assert.Equal("same", orderedWords[0].embedding.Label);
            Assert.Equal(0d, orderedWords[1].similarity, 8);
            Assert.Equal("close-ish", orderedWords[1].embedding.Label);
            Assert.Equal(-1d, orderedWords[2].similarity, 8);
            Assert.Equal("far", orderedWords[2].embedding.Label);
        }
    }
}
