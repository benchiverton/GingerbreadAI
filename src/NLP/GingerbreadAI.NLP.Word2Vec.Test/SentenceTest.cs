using System.IO;
using System.Text;
using Xunit;

namespace GingerbreadAI.NLP.Word2Vec.Test
{
    public class SentenceTest
    {
        [Fact]
        public void CanGetSentence()
        {
            long wordCount = 0;
            const string input = "This is a string. The String to test, the string   to prevail.\r\nWhat is the string?";
            var wordCollection = new WordCollection();
            wordCollection.AddWords(input, 11);
            wordCollection.InitWordPositions();
            const int maxSentenceLength = 50;
            var sentence = new long?[maxSentenceLength + 1];
            ulong nextRandom = 1;
            const float thresholdForOccurrenceOfWords = 0;
            long sentenceLength = 0;
            string[] lastLine = null;
            var reader = new StreamReader(new MemoryStream(Encoding.ASCII.GetBytes(input)));
            wordCount = NLP.Word2Vec.Word2Vec.SetSentence(reader, wordCount, sentence, ref nextRandom, ref sentenceLength,
                ref lastLine, wordCollection, thresholdForOccurrenceOfWords);

            Assert.Equal(16, wordCount);
            Assert.Equal(16, sentenceLength);
            Assert.NotNull(sentence[15]);
            Assert.Null(sentence[16]);
        }

        [Fact]
        public void WillSkipSentencesThatAreTooLong()
        {
            long wordCount = 0;
            const string input = "This is a string. The String to test, the strings   to prevail.\r\nWhat is the string?";
            var wordCollection = new WordCollection();
            wordCollection.AddWords(input, 11);
            wordCollection.InitWordPositions();
            var sentence = new long?[11];
            ulong nextRandom = 1;
            const float thresholdForOccurrenceOfWords = 0;
            long sentenceLength = 0;
            string[] lastLine = null;
            var reader = new StreamReader(new MemoryStream(Encoding.ASCII.GetBytes(input)));
            wordCount = NLP.Word2Vec.Word2Vec.SetSentence(reader, wordCount, sentence, ref nextRandom, ref sentenceLength,
                ref lastLine, wordCollection, thresholdForOccurrenceOfWords);

            Assert.Equal(4, wordCount);
            Assert.Equal(4, sentenceLength);
            Assert.NotNull(sentence[3]);
            Assert.Null(sentence[4]);
        }

        [Fact]
        public void DoesNotSufferFromOffByOne()
        {
            long wordCount = 0;
            const string input = "This is a string. The String to test, the strings   to prevail.\r\nWhat is the string?";
            var wordCollection = new WordCollection();
            wordCollection.AddWords(input, 11);
            wordCollection.InitWordPositions();
            var sentence = new long?[12];
            ulong nextRandom = 1;
            const float thresholdForOccurrenceOfWords = 0;
            long sentenceLength = 0;
            string[] lastLine = null;
            var reader = new StreamReader(new MemoryStream(Encoding.ASCII.GetBytes(input)));
            wordCount = NLP.Word2Vec.Word2Vec.SetSentence(reader, wordCount, sentence, ref nextRandom, ref sentenceLength,
                ref lastLine, wordCollection, thresholdForOccurrenceOfWords);

            Assert.Equal(12, wordCount);
            Assert.Equal(12, sentenceLength);
            Assert.NotNull(sentence[11]);
        }
    }
}
