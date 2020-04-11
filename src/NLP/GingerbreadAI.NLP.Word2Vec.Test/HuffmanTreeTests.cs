using Xunit;

namespace GingerbreadAI.NLP.Word2Vec.Test
{
    public class HuffmanTreeTests
    {

        [Fact]
        public void WillCreateAProperHuffmanTree()
        {
            var wordCollection = new WordCollection();
            AddWords(5, wordCollection, "f");
            AddWords(45, wordCollection, "a");
            AddWords(9, wordCollection, "e");
            AddWords(13, wordCollection, "b");
            AddWords(12, wordCollection, "c");
            AddWords(16, wordCollection, "d");
            wordCollection.InitWordPositions();
            new HuffmanTree().Create(wordCollection);

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
