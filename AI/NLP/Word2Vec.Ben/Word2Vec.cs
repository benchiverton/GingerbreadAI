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
        private const int ExpTableSize = 1000;
        private const int MaxCodeLength = 40;
        private const int MaxSentenceLength = 100;
        private const int TableSize = (int)1e8;
        private const int MinCount = 5;
        private const float ThresholdForOccurrenceOfWords = 1e-3f;

        private readonly int _numberOfIterations;
        private readonly int _numberOfDimensions;
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
            _numberOfDimensions = 50;
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

            WordCollection.RemoveWordsWithCountLessThanMinCount(MinCount);

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
            var iterationsRemaining = _numberOfIterations;
            var nextRandom = (ulong)id;
            var sum = WordCollection.GetTotalNumberOfWords();
            string[] lastLine = null;

            var skipGram = new SkipGram(_table, WordCollection.GetNumberOfUniqueWords(),
                Network.CloneNewWithWeightReferences(), WordCollection.GetTotalNumberOfWords(), _numberOfIterations, _numberOfThreads);

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
                        iterationsRemaining--;
                        if (iterationsRemaining == 0)
                            break;
                        wordCount = 0;
                        sentenceLength = 0;
                        reader.BaseStream.Seek(_fileSize / _numberOfThreads * id, SeekOrigin.Begin);
                        continue;
                    }
                    var wordIndex = sentence[sentencePosition];

                    nextRandom = nextRandom.LinearCongruentialGenerator();
                    nextRandom = skipGram.Train(sentencePosition, sentenceLength, sentence, wordIndex, nextRandom);

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
            if (lastLine != null && lastLine.Any())
            {
                loopEnd = HandleWords(reader, ref wordCount, sum, sentence, ref nextRandom, ref sentenceLength, lastLine, WordCollection);
                lastLine = null;
            }

            while (!loopEnd && (line = reader.ReadLine()) != null)
            {
                var words = splitRegex.Split(line);
                if (sentenceLength >= MaxSentenceLength - words.Length && words.Length < MaxSentenceLength)
                {
                    lastLine = words;
                    break;
                }
                loopEnd = HandleWords(reader, ref wordCount, sum, sentence, ref nextRandom, ref sentenceLength, words, WordCollection);
            }
            return wordCount;
        }

        private static bool HandleWords(StreamReader reader, ref long wordCount, long sum, long[] sentence, ref ulong nextRandom,
            ref long sentenceLength, IEnumerable<string> words, WordCollection wordCollection)
        {
            foreach (var word in words)
            {
                var wordIndex = wordCollection[word];
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
                if (ThresholdForOccurrenceOfWords > 0)
                {
                    var random = ((float)Math.Sqrt(wordCollection.GetOccuranceOfWord(word) / (ThresholdForOccurrenceOfWords * sum)) + 1) *
                              (ThresholdForOccurrenceOfWords * sum) / wordCollection.GetOccuranceOfWord(word);
                    nextRandom = nextRandom.LinearCongruentialGenerator();
                    if (random < (nextRandom & 0xFFFF) / (float)65536)
                        continue;
                }
                sentence[sentenceLength] = wordIndex.Value;
                sentenceLength++;
                if (sentenceLength >= MaxSentenceLength)
                {
                    return true;
                }
            }
            return false;
        }
    }
}