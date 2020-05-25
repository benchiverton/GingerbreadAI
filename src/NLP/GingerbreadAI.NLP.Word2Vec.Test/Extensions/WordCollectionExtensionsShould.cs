using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using GingerbreadAI.NLP.Word2Vec.Embeddings;
using GingerbreadAI.NLP.Word2Vec.Extensions;
using Xunit;

namespace GingerbreadAI.NLP.Word2Vec.Test.Extensions
{
    public class WordCollectionExtensionsShould
    {
        [Fact]
        public void ReturnCorrectUnigramTable()
        {
            var wordCollection = new WordCollection();
            AddWords(1, wordCollection, "a");
            AddWords(2, wordCollection, "b");
            AddWords(3, wordCollection, "c");
            AddWords(4, wordCollection, "d");
            AddWords(5, wordCollection, "e");
            wordCollection.InitWordPositions();

            var table = wordCollection.GetUnigramTable(30, 1);

            Assert.Equal(2, table.Count(x => x == 0));
            Assert.Equal(4, table.Count(x => x == 1));
            Assert.Equal(6, table.Count(x => x == 2));
            Assert.Equal(8, table.Count(x => x == 3));
            Assert.Equal(10, table.Count(x => x == 4));
        }

        [Fact]
        public void WriteAndReadFromStream()
        {
            var words = new string[] { "a", "b", "c" };
            var hiddenLayerWeights = new[,]{
                {0.11d, 0.12d},
                {0.21d, 0.22d},
                {0.31d, 0.32d}
            };
            using var memoryStream = new MemoryStream();
            using var writer = new StreamWriter(memoryStream, Encoding.UTF8);
            var wordEmbeddings = GetWordEmbeddings(words, 2, hiddenLayerWeights);
            wordEmbeddings.WriteEmbeddingToStream(writer);
            writer.Flush();
            memoryStream.Position = 0;
            using var reader = new StreamReader(memoryStream, Encoding.UTF8);
            var embedding = new List<WordEmbedding>();
            embedding.PopulateWordEmbeddingsFromStream(reader);

            Assert.Equal(3, embedding.Count);
            for (var wordIndex = 0; wordIndex < words.Length; wordIndex++)
            {
                Assert.Equal(words[wordIndex], embedding[wordIndex].Label);
                Assert.Equal(hiddenLayerWeights[wordIndex, 0], embedding[wordIndex].Vector[0]);
                Assert.Equal(hiddenLayerWeights[wordIndex, 1], embedding[wordIndex].Vector[1]);
            }
        }

        private static IEnumerable<WordEmbedding> GetWordEmbeddings(IReadOnlyList<string> words, int numberOfDimensions, double[,] hiddenLayerWeights)
        {
            var wordEmbeddings = new List<WordEmbedding>();
            for (var i = 0; i < words.Count; i++)
            {
                var vector = new List<double>();
                for (var dimensionIndex = 0; dimensionIndex < numberOfDimensions; dimensionIndex++)
                {
                    vector.Add(hiddenLayerWeights[i, dimensionIndex]);
                }

                wordEmbeddings.Add(new WordEmbedding(words[i], vector.ToArray()));
            }

            return wordEmbeddings;
        }

        [Fact]
        public void CreateAProperHuffmanTree()
        {
            var wordCollection = new WordCollection();
            AddWords(5, wordCollection, "f");
            AddWords(45, wordCollection, "a");
            AddWords(9, wordCollection, "e");
            AddWords(13, wordCollection, "b");
            AddWords(12, wordCollection, "c");
            AddWords(16, wordCollection, "d");
            wordCollection.InitWordPositions();
            wordCollection.CreateBinaryTree();

            var word = "a";
            var expectedCode = new[] { '\0', '\0', '\0', '\0' };
            var expectedPoint = new long[] { 4, 0, 0, 0 };
            var codeLength = 1;
            VerifyWordInfo(wordCollection, word, expectedCode, expectedPoint, codeLength);

            word = "c";
            expectedCode = new[] { (char)1, '\0', '\0', '\0' };
            expectedPoint = new long[] { 4, 3, 1, 0 };
            codeLength = 3;
            VerifyWordInfo(wordCollection, word, expectedCode, expectedPoint, codeLength);

            word = "b";
            expectedCode = new[] { (char)1, '\0', (char)1, '\0' };
            expectedPoint = new long[] { 4, 3, 1, 0 };
            codeLength = 3;
            VerifyWordInfo(wordCollection, word, expectedCode, expectedPoint, codeLength);

            word = "d";
            expectedCode = new[] { (char)1, (char)1, (char)1, '\0' };
            expectedPoint = new long[] { 4, 3, 2, 0 };
            codeLength = 3;
            VerifyWordInfo(wordCollection, word, expectedCode, expectedPoint, codeLength);

            word = "e";
            expectedCode = new[] { (char)1, (char)1, '\0', (char)1 };
            expectedPoint = new long[] { 4, 3, 2, 0 };
            codeLength = 4;
            VerifyWordInfo(wordCollection, word, expectedCode, expectedPoint, codeLength);

            word = "f";
            expectedCode = new[] { (char)1, (char)1, '\0', '\0' };
            expectedPoint = new long[] { 4, 3, 2, 0 };
            codeLength = 4;
            VerifyWordInfo(wordCollection, word, expectedCode, expectedPoint, codeLength);
        }

        private static void AddWords(int numberOfCopies, WordCollection wordCollection, string inputCharacter)
        {
            for (var i = 0; i < numberOfCopies; i++)
            {
                wordCollection.AddWords(inputCharacter, 4);
            }
        }

        private static void VerifyWordInfo(WordCollection wordCollection, string word, char[] expectedCode, long[] expectedPoints, int expectedCodeLength)
        {
            var position = wordCollection[word].Value;
            var code = wordCollection[position].Code;
            var points = wordCollection[position].Point;
            var codeLength = wordCollection[position].CodeLength;
            Assert.Equal(expectedCode, code);
            Assert.Equal(expectedPoints, points);
            Assert.Equal(expectedCodeLength, codeLength);
        }
    }
}
