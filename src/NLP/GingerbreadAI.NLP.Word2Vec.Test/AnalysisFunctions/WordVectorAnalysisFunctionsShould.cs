using System.Collections.Generic;
using System.Linq;
using GingerbreadAI.NLP.Word2Vec.AnalysisFunctions;
using Xunit;

namespace GingerbreadAI.NLP.Word2Vec.Test.AnalysisFunctions
{
    public class WordVectorAnalysisFunctionsShould
    {
        [Fact]
        public void CorrectlyGetMostSimilarWords()
        {
            var wordVectors = new List<(string word, List<double> vector)>
            {
                ("target", new List<double> {1d, 1d}),
                ("far", new List<double> {-1d, -1d}),
                ("close-ish", new List<double> {-1d, 1d}),
                ("same", new List<double> {1d, 1d}),
            };

            var orderedWords = WordVectorAnalysisFunctions.GetMostSimilarWords("target", wordVectors, 3).ToArray();
            Assert.Equal(1d, orderedWords[0].similarity, 8);
            Assert.Equal("same", orderedWords[0].word);
            Assert.Equal(0d, orderedWords[1].similarity, 8);
            Assert.Equal("close-ish", orderedWords[1].word);
            Assert.Equal(-1d, orderedWords[2].similarity, 8);
            Assert.Equal("far", orderedWords[2].word);
        }
    }
}
