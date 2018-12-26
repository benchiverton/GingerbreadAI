using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using NeuralNetwork;
using NeuralNetwork.Data;
using NeuralNetwork.Library.Extensions;
using Word2Vec.Ben.Extensions;

namespace Word2Vec.Ben
{
    //C# Word2Vec based on https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c
    //Any comments in here is to show how misleading they can be. :)
    public class Word2Vec
    {
        private const int MaxCodeLength = 40;
        private const int MaxSentenceLength = 3000;
        private const int TableSize = (int)1e8;
        private const int MinCount = 5;
        private const float ThresholdForOccurrenceOfWords = 1e-3f;
        private const int NumberOfDimensions = 100;

        private readonly int _numberOfIterations;
        private readonly int _numberOfThreads;
        private readonly string _saveVocabFile;
        private readonly string _trainFile;

        private long _fileSize;
        private int[] _table;

        public WordCollection WordCollection { get; private set; }
        public Layer Network { get; private set; }

        public Word2Vec(string trainFileName, string saveVocabFileName, int numberOfIterations, int numberOfThreads)
        {
            _trainFile = trainFileName;
            _saveVocabFile = saveVocabFileName;
            _numberOfIterations = numberOfIterations;
            _numberOfThreads = numberOfThreads;
        }

        public void TrainModel()
        {
            Setup();
            Train();
            GC.Collect();
        }

        private void Setup()
        {
            _fileSize = new FileInfo(_trainFile).Length;

            WordCollection = new WordCollection();
            FileHandler.GetWordDictionaryFromFile(_trainFile, WordCollection, MaxCodeLength);
            WordCollection.RemoveWordsWithCountLessThanMinCount(MinCount);
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
            var inputLayer = new Layer("Input", WordCollection.GetNumberOfUniqueWords(), new Layer[] { });
            var hiddenLayer = new Layer("Hidden", NumberOfDimensions, new Layer[] { inputLayer });
            Network = new Layer("Output", WordCollection.GetNumberOfUniqueWords(), new Layer[] { hiddenLayer });
            // we only want to randomise weights for the hidden layer (?)
            LayerInitialiser.Initialise(new Random(), hiddenLayer);

            HuffmanTree.Create(WordCollection, MaxCodeLength);

            GC.Collect();
        }

        private void InitUnigramTable()
        {
            var vocabSize = WordCollection.GetNumberOfUniqueWords();
            if (vocabSize == 0) return;

            var power = 0.75;
            var trainWordsPow = WordCollection.GetTrainWordsPow(power);
            var keys = WordCollection.GetWords().ToArray();
            var d1 = Math.Pow(WordCollection.GetOccuranceOfWord(keys[0]), power) / trainWordsPow;

            var i = 0;
            _table = new int[TableSize];
            for (var a = 0; a < TableSize; a++)
            {
                _table[a] = i;
                if (a / (double)TableSize > d1)
                {
                    i++;
                    d1 += Math.Pow(WordCollection.GetOccuranceOfWord(keys[i]), power) / trainWordsPow;
                }
                if (i >= vocabSize)
                {
                    i = vocabSize - 1;
                }
            }
        }

        private void TrainModelThreadStart(int id)
        {
            var splitRegex = new Regex("\\s");
            long sentenceLength = 0;
            long sentencePosition = 0;
            long wordCount = 0;
            var sentence = new long?[MaxSentenceLength]; //Sentence elements will not be null to my understanding
            var localIterations = _numberOfIterations;
            var nextRandom = (ulong)id;
            var sum = WordCollection.GetTotalNumberOfWords();
            string[] lastLine = null;

            var skipGram = new SkipGram(_table, Network.CloneNewWithWeightReferences(),
                WordCollection.GetTotalNumberOfWords(), _numberOfIterations, _numberOfThreads);

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
                        localIterations--;
                        if (localIterations == 0)
                            break;
                        wordCount = 0;
                        sentenceLength = 0;
                        reader.BaseStream.Seek(_fileSize / _numberOfThreads * id, SeekOrigin.Begin);
                        Console.WriteLine($"Iterations remaining: {localIterations} Thread: {id}");
                        continue;
                    }
                    var wordIndex = sentence[sentencePosition];
                    if (!wordIndex.HasValue) continue;

                    nextRandom = skipGram.Train(sentencePosition, sentenceLength, sentence, wordIndex.Value, nextRandom);

                    sentencePosition++;
                    if (sentencePosition >= sentenceLength) sentenceLength = 0;
                }
            }
            GC.Collect();
        }

        private long SetSentence(StreamReader reader, Regex splitRegex, long wordCount, long sum, long?[] sentence,
            ref ulong nextRandom, ref long sentenceLength, ref string[] lastLine)
        {
            string line;
            var loopEnd = false;
            var numberOfLineRead = 0;

            if (lastLine != null && lastLine.Any())
            {
                loopEnd = HandleWords(ref wordCount, sum, sentence, ref nextRandom, ref sentenceLength, lastLine);
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
                loopEnd = HandleWords(ref wordCount, sum, sentence, ref nextRandom, ref sentenceLength, words);
            }
            return wordCount;
        }

        private bool HandleWords(ref long wordCount, long sum, long?[] sentence, ref ulong nextRandom,
            ref long sentenceLength, IEnumerable<string> words)
        {
            var loopEnd = false;
            foreach (var word in words)
            {
                var wordIndex = WordCollection[word];
                if (!wordIndex.HasValue) continue;

                wordCount++;
                if (ThresholdForOccurrenceOfWords > 0)
                {
                    var ran = ((float)Math.Sqrt(WordCollection.GetOccuranceOfWord(word) / (ThresholdForOccurrenceOfWords * sum)) + 1)
                        * (ThresholdForOccurrenceOfWords * sum) / WordCollection.GetOccuranceOfWord(word);
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