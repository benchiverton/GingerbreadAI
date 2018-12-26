using NegativeSampling;
using NeuralNetwork.Data;
using System;
using Word2Vec.Ben.Extensions;

namespace Word2Vec.Ben
{
    public class SkipGram
    {
        private readonly int _windowSize;
        private readonly int _negativeSamples;
        private readonly int[] _table;
        private readonly long _totalWords;
        private readonly NegativeSampler _negativeSampler;

        public SkipGram(int[] table, Layer network, long totalWords, int iterations, int threads)
        {
            _windowSize = 5;
            _negativeSamples = 5;
            _table = table;
            _totalWords = totalWords;

            var negativeSampler = new NegativeSampler(network, 0.025, learningRateModifier: LearningAction);

            _negativeSampler = negativeSampler;
        }

        public ulong Train(long sentencePosition, long sentenceLength, long?[] sentence, long targetWord, ulong nextRandom)
        {
            nextRandom = nextRandom.LinearCongruentialGenerator();
            var randomWindowPosition = (long)(nextRandom % (ulong)_windowSize);
            for (var offsetWithinWindow = randomWindowPosition; offsetWithinWindow < _windowSize * 2 + 1 - randomWindowPosition; offsetWithinWindow++)
            {
                if (offsetWithinWindow == _windowSize) continue;

                var indexOfCurrentContextWordInSentence = sentencePosition - _windowSize + offsetWithinWindow;
                if (indexOfCurrentContextWordInSentence < 0 || indexOfCurrentContextWordInSentence >= sentenceLength) continue;

                var indexOfContextWord = sentence[indexOfCurrentContextWordInSentence];
                if (!indexOfContextWord.HasValue) continue;

                nextRandom = NegativeSampling(indexOfContextWord.Value, targetWord, nextRandom);
            }

            return nextRandom;
        }

        private ulong NegativeSampling(long indexOfContextWord, long targetWord, ulong nextRandom)
        {
            // this is the positive sample
            _negativeSampler.NegativeSample((int)indexOfContextWord, (int)targetWord, true);

            for (var i = 0; i < _negativeSamples; i++)
            {
                var randomTarget = SelectTarget(ref nextRandom);
                if (randomTarget == targetWord) continue; // don't want to use positive sample as negative sample

                _negativeSampler.NegativeSample((int)indexOfContextWord, (int)randomTarget, false);
            }

            return nextRandom;
        }

        private double LearningAction(double alpha)
            => alpha <= 0.0001 ? 0.0001 : alpha * _totalWords / (_totalWords + 1);

        private long SelectTarget(ref ulong nextRandom)
        {
            nextRandom = nextRandom.LinearCongruentialGenerator();
            long target = _table[(nextRandom >> 16) % (ulong)_table.Length];
            return target;
        }
    }
}
