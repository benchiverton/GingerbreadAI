using BackPropagation;
using NeuralNetwork.Data;
using System;
using System.Collections.Generic;
using System.Text;
using Word2Vec.Extensions;

namespace Word2Vec
{
    public class SkipGram
    {
        private readonly int _windowSize;
        private readonly int _numberOfWords;
        private readonly int _negativeSamples;
        private readonly int[] _table;
        private readonly int _maxExp;
        private readonly float[] _expTable;
        private readonly BackPropagator _backPropagator;

        public SkipGram(int[] table, int maxExp, float[] expTable, int numberOfWords, Layer network, int iterations)
        {
            _windowSize = 5;
            _negativeSamples = 5;
            _table = table;
            _maxExp = maxExp;
            _expTable = expTable;
            _numberOfWords = numberOfWords;

            var backPropagator = new BackPropagator(network, 0.1, momentum:0.5, learningAction:(i) => 1 - (i / (numberOfWords * iterations)));

            _backPropagator = backPropagator;
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

                wordsWithinWindow.Add(sentence[wordWithinWindow]);
            }

            nextRandom = NegativeSampling(targetWord, wordsWithinWindow, nextRandom);

            return nextRandom;
        }

        private ulong NegativeSampling(long targetWord, List<long> wordsWithinWindow, ulong nextRandom)
        {
            var inputs = new double[_numberOfWords];
            for (var i = 0; i < inputs.Length; i++)
            {
                inputs[i] = int.MinValue;
            }
            foreach (var word in wordsWithinWindow)
            {
                inputs[word] = int.MaxValue;
            }

            var targetOutput = new double?[_numberOfWords];
            targetOutput[targetWord] = 1;
            for (int i = 0; i < _negativeSamples - 1; i++)
            {
                var randomTarget = SelectTarget(ref nextRandom);
                if (randomTarget == targetWord) continue; // don't want to override target
                targetOutput[randomTarget] = 0;
            }

            _backPropagator.BackPropagate(inputs, targetOutput);

            return nextRandom;
        }

        private long SelectTarget(ref ulong nextRandom)
        {
            nextRandom = nextRandom.LinearCongruentialGenerator();
            long target = _table[(nextRandom >> 16) % (ulong)_table.Length];
            if (target == 0) target = (long)(nextRandom % (ulong)(_numberOfWords - 1) + 1);
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
