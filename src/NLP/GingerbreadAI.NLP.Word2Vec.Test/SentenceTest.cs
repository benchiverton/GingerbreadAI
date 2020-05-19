using System;
using System.IO;
using System.Text;
using Xunit;

namespace GingerbreadAI.NLP.Word2Vec.Test
{
    public class Word2VecTrainerShould
    {
        [Fact]
        public void CorrectlyGetSentence()
        {
            const string input = "This is a string. The String to test, the string   to prevail.\r\nWhat is the string?";
            var wordCollection = new WordCollection();
            wordCollection.AddWords(input, 11);
            wordCollection.InitWordPositions();
            const int maxSentenceLength = 50;
            var sentence = new int?[maxSentenceLength + 1];
            var nextRandom = new Random();
            const double thresholdForOccurrenceOfWords = 0;
            var sentenceLength = 0;
            string[] lastLine = null;
            var reader = new StreamReader(new MemoryStream(Encoding.ASCII.GetBytes(input)));
            Word2VecTrainer.SetSentence(wordCollection, reader, sentence, nextRandom, ref sentenceLength, ref lastLine, thresholdForOccurrenceOfWords);
            reader.Dispose();

            Assert.Equal(16, sentenceLength);
            Assert.NotNull(sentence[15]);
            Assert.Null(sentence[16]);
        }

        [Fact]
        public void SkipSentencesThatAreTooLong()
        {
            const string input = "This is a string. The String to test, the strings   to prevail.\r\nWhat is the string?";
            var wordCollection = new WordCollection();
            wordCollection.AddWords(input, 11);
            wordCollection.InitWordPositions();
            var sentence = new int?[11];
            var nextRandom = new Random();
            const double thresholdForOccurrenceOfWords = 0;
            var sentenceLength = 0;
            string[] lastLine = null;
            var reader = new StreamReader(new MemoryStream(Encoding.ASCII.GetBytes(input)));
            Word2VecTrainer.SetSentence(wordCollection, reader, sentence, nextRandom, ref sentenceLength, ref lastLine, thresholdForOccurrenceOfWords);
            reader.Dispose();

            Assert.Equal(4, sentenceLength);
            Assert.NotNull(sentence[3]);
            Assert.Null(sentence[4]);
        }

        [Fact]
        public void NotSufferFromOffByOne()
        {
            const string input = "This is a string. The String to test, the strings   to prevail.\r\nWhat is the string?";
            var wordCollection = new WordCollection();
            wordCollection.AddWords(input, 11);
            wordCollection.InitWordPositions();
            var sentence = new int?[12];
            var nextRandom = new Random();
            const double thresholdForOccurrenceOfWords = 0;
            var sentenceLength = 0;
            string[] lastLine = null;
            var reader = new StreamReader(new MemoryStream(Encoding.ASCII.GetBytes(input)));
            Word2VecTrainer.SetSentence(wordCollection, reader, sentence, nextRandom, ref sentenceLength, ref lastLine, thresholdForOccurrenceOfWords);
            reader.Dispose();

            Assert.Equal(12, sentenceLength);
            Assert.NotNull(sentence[11]);
        }
    }
}
