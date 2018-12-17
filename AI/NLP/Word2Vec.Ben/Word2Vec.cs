﻿using NeuralNetwork;
using NeuralNetwork.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Word2Vec.Extensions;

namespace Word2Vec
{
    //C# Word2Vec based on https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c
    //Any comments in here is to show how misleading they can be. :)
    public class Word2Vec
    {
        private const int ExpTableSize = 1000;
        private const int MaxCodeLength = 40;
        private const int MaxExp = 6;
        private const int MaxSentenceLength = 10000;
        private const int TableSize = (int)1e8;
        private readonly float[] _expTable;
        private readonly int _numberOfIterations;
        private readonly int _numberOfDimensions;
        private readonly int _minCount = 5;
        private readonly int _numberOfThreads;
        private readonly float _thresholdForOccurrenceOfWords = 1e-3f;
        private readonly string _saveVocabFile;
        private readonly string _trainFile;
        private long _fileSize;
        private int[] _table;
        private SkipGram _skipGram;

        public WordCollection WordCollection { get; private set; }
        public Layer Network { get; private set; }

        public Word2Vec(string trainFileName, string saveVocabFileName, int numberOfIterations)
        {
            _trainFile = trainFileName;
            _saveVocabFile = saveVocabFileName;
            _numberOfIterations = numberOfIterations;
            _expTable = new float[ExpTableSize + 1];
            _numberOfThreads = 1;
            _numberOfDimensions = 35;
            for (var i = 0; i < ExpTableSize; i++)
            {
                _expTable[i] = (float)Math.Exp((i / (float)ExpTableSize * 2 - 1) * MaxExp);
                _expTable[i] = _expTable[i] / (_expTable[i] + 1);
            }
        }

        public void TrainModel()
        {
            Setup();
            Train();
            GC.Collect();
        }

        private void Setup()
        {
            WordCollection = new WordCollection();
            _fileSize = new FileInfo(_trainFile).Length;

            FileHandler.GetWordDictionaryFromFile(_trainFile, WordCollection, MaxCodeLength);

            WordCollection.RemoveWordsWithCountLessThanMinCount(_minCount);

            if (!string.IsNullOrEmpty(_saveVocabFile))
                FileHandler.SaveWordDictionary(_saveVocabFile, WordCollection);

            InitUnigramTable();

            InitNetwork();
        }

        private void Train()
        {
            var parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = _numberOfThreads
            };
            WordCollection.InitWordPositions();
            var result = Parallel.For(0, _numberOfThreads, parallelOptions, TrainModelThreadStart);
            if (!result.IsCompleted)
                throw new InvalidOperationException();
        }

        private void InitNetwork()
        {
            HuffmanTree.Create(WordCollection, MaxCodeLength);

            var inputLayer = new Layer("Input", WordCollection.GetNumberOfUniqueWords(), new Layer[] { });
            var hiddenLayer = new Layer("Hidden", _numberOfDimensions, new Layer[] { inputLayer });
            Network = new Layer("Output", WordCollection.GetNumberOfUniqueWords(), new Layer[] { hiddenLayer });

            LayerInitialiser.Initialise(new Random(), Network);

            _skipGram = new SkipGram(_table, MaxExp, _expTable, WordCollection.GetNumberOfUniqueWords(), Network, _numberOfIterations);

            GC.Collect();
        }

        private void InitUnigramTable()
        {
            if (WordCollection.GetNumberOfUniqueWords() == 0)
                return;

            int a;
            var power = 0.75;
            _table = new int[TableSize];
            var trainWordsPow = WordCollection.GetTrainWordsPow(power);

            var i = 0;
            var keys = WordCollection.GetWords().ToArray();
            var d1 = Math.Pow(WordCollection.GetOccuranceOfWord(keys.First()), power) / trainWordsPow;
            for (a = 0; a < TableSize; a++)
            {
                _table[a] = i;
                if (a / (double)TableSize > d1)
                {
                    i++;
                    d1 += Math.Pow(WordCollection.GetOccuranceOfWord(keys[i]), power) / trainWordsPow;
                }
                if (i >= WordCollection.GetNumberOfUniqueWords())
                {
                    i = WordCollection.GetNumberOfUniqueWords() - 1;
                }
            }
        }

        private void TrainModelThreadStart(int id)
        {
            var splitRegex = new Regex("\\s");
            long sentenceLength = 0;
            long sentencePosition = 0;
            long wordCount = 0;
            var sentence = new long[MaxSentenceLength + 1]; //Sentence elements will not be null to my understanding
            var localIter = _numberOfIterations;

            var nextRandom = (ulong)id;
            var sum = WordCollection.GetTotalNumberOfWords();
            string[] lastLine = null;
            using (var reader = File.OpenText(_trainFile))
            {
                reader.BaseStream.Seek(_fileSize / _numberOfThreads * id, SeekOrigin.Begin);
                while (true)
                {
                    if (sentenceLength == 0)
                    {
                        wordCount = SetSentence(reader, splitRegex, wordCount, sum, sentence, ref nextRandom, ref sentenceLength, ref lastLine);
                        sentencePosition = 0;
                    }
                    if (reader.EndOfStream || wordCount > sum / _numberOfThreads)
                    {
                        localIter--;
                        if (localIter == 0)
                            break;
                        wordCount = 0;
                        sentenceLength = 0;
                        reader.BaseStream.Seek(_fileSize / _numberOfThreads * id, SeekOrigin.Begin);
                        Console.WriteLine($"Iterations remaining: {localIter} Thread: {id}");
                        continue;
                    }
                    var wordIndex = sentence[sentencePosition];

                    nextRandom = nextRandom.LinearCongruentialGenerator();
                    nextRandom = _skipGram.Train(sentencePosition, sentenceLength, sentence, wordIndex, nextRandom);
                    sentencePosition++;

                    if (sentencePosition >= sentenceLength)
                    {
                        sentenceLength = 0;
                    }
                }
            }
            GC.Collect();
        }

        private long SetSentence(StreamReader reader, Regex splitRegex, long wordCount, long sum, long[] sentence,
            ref ulong nextRandom, ref long sentenceLength, ref string[] lastLine)
        {
            string line;
            var loopEnd = false;
            var numberOfLineRead = 0;

            if (lastLine != null && lastLine.Any())
            {
                loopEnd = HandleWords(reader, ref wordCount, sum, sentence, ref nextRandom, ref sentenceLength, lastLine);
                lastLine = null;
            }

            while (!loopEnd && (line = reader.ReadLine()) != null)
            {
                numberOfLineRead++;
                var words = splitRegex.Split(line);
                if (sentenceLength >= MaxSentenceLength - words.Length && words.Length < MaxSentenceLength)
                {
                    lastLine = words;
                    break;
                }
                loopEnd = HandleWords(reader, ref wordCount, sum, sentence, ref nextRandom, ref sentenceLength, words);
            }
            return wordCount;
        }

        private bool HandleWords(StreamReader reader, ref long wordCount, long sum, long[] sentence, ref ulong nextRandom,
            ref long sentenceLength, IEnumerable<string> words)
        {
            var loopEnd = false;
            foreach (var word in words)
            {
                var wordIndex = WordCollection[word];
                if (reader.EndOfStream)
                {
                    return true;
                }
                if (!wordIndex.HasValue)
                    continue;
                wordCount++;
                if (wordIndex == 0)
                {
                    return true;
                }
                if (_thresholdForOccurrenceOfWords > 0)
                {
                    var ran = ((float)Math.Sqrt(WordCollection.GetOccuranceOfWord(word) /
                                                 (_thresholdForOccurrenceOfWords * sum)) + 1) *
                              (_thresholdForOccurrenceOfWords * sum) / WordCollection.GetOccuranceOfWord(word);
                    nextRandom = nextRandom.LinearCongruentialGenerator();
                    if (ran < (nextRandom & 0xFFFF) / (float)65536)
                        continue;
                }
                sentence[sentenceLength] = wordIndex.Value;
                sentenceLength++;
                if (sentenceLength >= MaxSentenceLength)
                {
                    return true;
                }
            }
            return loopEnd;
        }
    }
}