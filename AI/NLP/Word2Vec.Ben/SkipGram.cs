using System;
using System.Collections.Generic;
using System.Linq;
using NegativeSampling;
using NeuralNetwork.Data;
using Word2Vec.Ben.Extensions;

namespace Word2Vec.Ben
{
    public class SkipGram
    {
        private readonly int _windowSize;
        private readonly int _numberOfUniqueWords;
        private readonly int _negativeSamples;
        private readonly int[] _table;
        private readonly NegativeSampler _negativeSampler;

        public SkipGram(int[] table, int numberOfUniqueWords, Layer network, long totalWords, int iterations, int threads)
        {
            _windowSize = 5;
            _negativeSamples = 2;
            _table = table;
            _numberOfUniqueWords = numberOfUniqueWords;

            var negativeSampler = new NegativeSampler(network, 0.25, learningAction: (i) => i < 0.001 ? 0.001 : i * (double)totalWords/(totalWords+1));

            _negativeSampler = negativeSampler;
        }

        public ulong Train(long sentencePosition, long sentenceLength, long[] sentence, long targetWord, ulong nextRandom)
        {
            var wordsWithinWindow = new List<long>();

            var randomWindowPosition = (long)(nextRandom % (ulong)_windowSize);
            for (var offsetWithinWindow = randomWindowPosition; offsetWithinWindow < _windowSize * 2 + 1 - randomWindowPosition; offsetWithinWindow++)
            {
                if (offsetWithinWindow == _windowSize) continue;

                var indexOfCurrentContextWordInSentence = sentencePosition - _windowSize + offsetWithinWindow;
                if (indexOfCurrentContextWordInSentence < 0 || indexOfCurrentContextWordInSentence >= sentenceLength)
                    continue;
                var wordWithinWindow = sentence[indexOfCurrentContextWordInSentence];

                wordsWithinWindow.Add(wordWithinWindow);
            }

            nextRandom = NegativeSampling(targetWord, wordsWithinWindow, nextRandom);

            return nextRandom;
        }

        private ulong NegativeSampling(long targetWord, List<long> wordsWithinWindow, ulong nextRandom)
        {
            foreach (var word in wordsWithinWindow)
            {
                _negativeSampler.NegativeSample((int)word, (int)targetWord, true);

                for (var i = 0; i < _negativeSamples - 1; i++)
                {
                    var randomTarget = SelectTarget(ref nextRandom);
                    if (randomTarget == targetWord) continue; // don't want to override target

                    _negativeSampler.NegativeSample((int)word, (int)randomTarget, false);
                }
            }

            return nextRandom;
        }

        private long SelectTarget(ref ulong nextRandom)
        {
            nextRandom = nextRandom.LinearCongruentialGenerator();
            long target = _table[(nextRandom >> 16) % (ulong)_table.Length];
            if (target == 0) target = (long)(nextRandom % (ulong)(_numberOfUniqueWords - 1) + 1);
            return target;
        }

        private static float[] InitOutputErrorContainer(int numberOfDimensions)
        {
            var accumulatedOutputError = new float[numberOfDimensions];
            for (var dimensionIndex = 0; dimensionIndex < numberOfDimensions; dimensionIndex++)
                accumulatedOutputError[dimensionIndex] = 0;
            return accumulatedOutputError;
        }
    }
}
