using System.Collections.Generic;
using System.Linq;
using GingerbreadAI.NLP.Word2Vec.SimilarityFunctions;

namespace GingerbreadAI.NLP.Word2Vec.AnalysisFunctions
{
    public static class WordVectorAnalysisFunctions
    {
        /// <summary>
        /// Returns the topn most similar words to the one given.
        /// </summary>
        public static IEnumerable<(string word, double similarity)> GetMostSimilarWords(string word, IEnumerable<(string word, List<double> vector)> wordVectors, 
            int topn = 10, SimilarityFunctionType similarityFunctionType = SimilarityFunctionType.Cosine)
        {
            var similarityFunction = SimilarityFunctionResolver.ResolveSimilarityFunction(similarityFunctionType);

            var wordVectorArray = wordVectors.ToArray();
            var wordVector = wordVectorArray.First(wv => wv.word == word);

            return wordVectorArray.Where(wv => wv.word != word)
                .Select(otherWordVector => (otherWordVector.word, similarityFunction.Invoke(wordVector.vector.ToArray(), otherWordVector.vector.ToArray())))
                .OrderByDescending(owcs => owcs.Item2)
                .Take(topn);
        }
    }
}
