using NegativeSampling;
using NeuralNetwork;
using NeuralNetwork.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace Word2Vec
{
    //C# Word2Vec based on https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c
    public class Word2VecUsingLibrary
    {
        private const int MaxCodeLength = 40;
        private readonly int _maxSentenceLength;
        private const int TableSize = (int)1e8;
        private readonly int _numberOfIterations;
        private readonly int _numberOfDimensions;
        private readonly int _minCount = 5;
        private readonly int _negative;
        private readonly int _numberOfThreads = 4;
        private readonly float _thresholdForOccurrenceOfWords = 1e-3f;
        private readonly int _windowSize;
        private readonly FileHandler _fileHandler;
        private readonly WordCollection _wordCollection = new WordCollection();

        private double _alpha;
        private double _startingAlpha;
        private int[] _table;
        private int _wordCountActual;
        private Layer _neuralNetwork;

        public Word2VecUsingLibrary(
            string trainFileName,
            string outputFileName,
            int numberOfThreads = 4,
            int numberOfIterations = 4,
            int negative = 5,
            int numberOfDimensions = 50,
            int windowSize = 5,
            float alpha = 0.025f,
            int maxSentenceLength = 10000,
            float thresholdForOccurrenceOfWords = 1e-3f
        )
        {
            _numberOfThreads = numberOfThreads;
            _numberOfIterations = numberOfIterations;
            _negative = negative;
            _numberOfDimensions = numberOfDimensions;
            _windowSize = windowSize;
            _alpha = alpha;
            _fileHandler = new FileHandler(trainFileName, outputFileName);
            _maxSentenceLength = maxSentenceLength;
            _thresholdForOccurrenceOfWords = thresholdForOccurrenceOfWords;
        }

        public void TrainModel()
        {
            Setup();
            Train();
            _fileHandler.WriteOutputMatrix(_wordCollection, _neuralNetwork);
            GC.Collect();
        }

        private void Setup()
        {
            _startingAlpha = _alpha;

            _fileHandler.GetWordDictionaryFromFile(_wordCollection, MaxCodeLength);

            _wordCollection.RemoveWordsWithCountLessThanMinCount(_minCount);

            InitNetwork();

            if (_negative > 0)
            {
                InitUnigramTable();
            }
        }

        private void Train()
        {
            var parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = _numberOfThreads
            };
            _wordCollection.InitWordPositions();
            var result = Parallel.For(0, _numberOfThreads, parallelOptions, TrainModelThreadStart);
            if (!result.IsCompleted)
            {
                throw new InvalidOperationException();
            }
        }

        private void InitNetwork()
        {
            var inputLayer = new Layer("input", _wordCollection.GetNumberOfUniqueWords(), new Layer[0]);
            var hiddenLayer = new Layer("hidden", _numberOfDimensions, new[] { inputLayer });
            _neuralNetwork = new Layer("output", _wordCollection.GetNumberOfUniqueWords(), new[] { hiddenLayer });
            // do not initialise output weights, as above (?)
            hiddenLayer.Initialise(new Random());

            var huffmanTree = new HuffmanTree();
            huffmanTree.Create(_wordCollection);
            GC.Collect();
        }

        private void InitUnigramTable()
        {
            if (_wordCollection.GetNumberOfUniqueWords() == 0)
            {
                return;
            }

            int a;
            var power = 0.75;
            _table = new int[TableSize];
            var trainWordsPow = _wordCollection.GetTrainWordsPow(power);

            var i = 0;
            var keys = _wordCollection.GetWords().ToArray();
            var d1 = Math.Pow(_wordCollection.GetOccurrenceOfWord(keys.First()), power) / trainWordsPow;
            for (a = 0; a < TableSize; a++)
            {
                _table[a] = i;
                if (a / (double)TableSize > d1)
                {
                    i++;
                    d1 += Math.Pow(_wordCollection.GetOccurrenceOfWord(keys[i]), power) / trainWordsPow;
                }
                if (i >= _wordCollection.GetNumberOfUniqueWords())
                {
                    i = _wordCollection.GetNumberOfUniqueWords() - 1;
                }
            }
        }

        private void TrainModelThreadStart(int id)
        {
            int sentenceLength = 0;
            int sentencePosition = 0;
            int wordCount = 0, lastWordCount = 0;
            var sentence = new int?[_maxSentenceLength];
            var localIter = _numberOfIterations;
            var localNetwork = _neuralNetwork.CloneWithNodeAndWeightReferences();

            var random = new Random();
            var sum = _wordCollection.GetTotalNumberOfWords();
            string[] lastLine = null;
            using (var reader = _fileHandler.GetReader())
            {
                reader.BaseStream.Seek(_fileHandler.FileSize / _numberOfThreads * id, SeekOrigin.Begin);
                while (true)
                {
                    if (wordCount - lastWordCount > 10000)
                    {
                        _wordCountActual += wordCount - lastWordCount;
                        lastWordCount = wordCount;
                        _alpha = _startingAlpha * (1 - _wordCountActual / (float)(_numberOfIterations * sum + 1));
                        if (_alpha < _startingAlpha * (float)0.0001)
                        {
                            _alpha = _startingAlpha * (float)0.0001;
                        }
                    }
                    if (sentenceLength == 0)
                    {
                        wordCount = SetSentence(reader, wordCount, sentence, random, ref sentenceLength, ref lastLine);
                        sentencePosition = 0;
                    }
                    if (reader.EndOfStream || wordCount > sum / _numberOfThreads)
                    {
                        _wordCountActual += wordCount - lastWordCount;
                        localIter--;
                        if (localIter == 0)
                        {
                            break;
                        }
                        wordCount = 0;
                        lastWordCount = 0;
                        sentenceLength = 0;
                        reader.BaseStream.Seek(_fileHandler.FileSize / _numberOfThreads * id, SeekOrigin.Begin);
                        Console.WriteLine($"Iterations remaining: {localIter} Thread: {id}");
                        continue;
                    }
                    var wordIndex = sentence[sentencePosition];
                    if (!wordIndex.HasValue)
                    {
                        continue;
                    }

                    SkipGram(localNetwork, sentencePosition, sentenceLength, sentence, wordIndex.Value, random);

                    sentencePosition++;

                    if (sentencePosition >= sentenceLength)
                    {
                        sentenceLength = 0;
                    }
                }
            }
            GC.Collect();
        }

        public int SetSentence(StreamReader reader, int wordCount, int?[] sentence, Random random, ref int sentenceLength,
            ref string[] lineThatGotCutOff)
        {
            string line;
            var loopEnd = false;
            if (lineThatGotCutOff != null && lineThatGotCutOff.Any())
            {
                loopEnd = HandleWords(reader, ref wordCount, sentence, random, ref sentenceLength, lineThatGotCutOff);
                lineThatGotCutOff = null;
            }

            while (!loopEnd && (line = reader.ReadLine()) != null)
            {
                var words = WordCollection.ParseWords(line).Select(WordCollection.Clean).ToArray();
                if (words.Length > sentence.Length)
                {
                    continue;
                }
                if (sentenceLength > sentence.Length - words.Length)
                {
                    lineThatGotCutOff = words;
                    break;
                }
                loopEnd = HandleWords(reader, ref wordCount, sentence, random, ref sentenceLength, words);
            }
            return wordCount;
        }

        private bool HandleWords(StreamReader reader, ref int wordCount, int?[] sentence, Random random,
            ref int sentenceLength, IEnumerable<string> words)
        {
            var totalNumberOfWords = _wordCollection.GetTotalNumberOfWords();
            foreach (var word in words)
            {
                var wordIndex = _wordCollection[word];
                if (!wordIndex.HasValue)
                {
                    continue;
                }
                wordCount++;

                //Subsampling of frequent words
                if (_thresholdForOccurrenceOfWords > 0)
                {
                    var something = ((float)Math.Sqrt(_wordCollection.GetOccurrenceOfWord(word) / (_thresholdForOccurrenceOfWords * totalNumberOfWords)) + 1)
                                    * (_thresholdForOccurrenceOfWords * totalNumberOfWords) / _wordCollection.GetOccurrenceOfWord(word);
                    if (something < (random.Next() & 0xFFFF) / (float)65536)
                    {
                        continue;
                    }
                }
                sentence[sentenceLength] = (int)wordIndex.Value;
                sentenceLength++;
                if (sentenceLength > sentence.Length)
                {
                    return true;
                }
            }
            if (reader.EndOfStream)
            {
                return true;
            }
            return false;
        }

        private void SkipGram(Layer neuralNetwork, int sentencePosition, int sentenceLength, int?[] sentence, int indexOfCurrentWord, Random random)
        {
            var randomWindowPosition = (int)(random.Next() % (uint)_windowSize);

            for (var offsetWithinWindow = randomWindowPosition; offsetWithinWindow < _windowSize * 2 + 1 - randomWindowPosition; offsetWithinWindow++)
            {
                if (offsetWithinWindow != _windowSize)
                {
                    var indexOfCurrentContextWordInSentence = sentencePosition - _windowSize + offsetWithinWindow;

                    if (indexOfCurrentContextWordInSentence >= 0 && indexOfCurrentContextWordInSentence < sentenceLength)
                    {
                        var indexOfContextWord = sentence[indexOfCurrentContextWordInSentence];

                        if (indexOfContextWord.HasValue)
                        {
                            NegativeSampling(neuralNetwork, (int)indexOfCurrentWord, (int)indexOfContextWord, random);
                        }
                    }
                }
            }
        }

        private void NegativeSampling(Layer neuralNetwork, int indexOfCurrentWord, int indexOfContextWord, Random random)
        {
            for (var i = 0; i < _negative; i++)
            {
                bool isPositiveSample;
                int target;
                if (i == 0)
                {
                    target = indexOfContextWord;
                    isPositiveSample = true;
                }
                else
                {
                    target = _table[random.Next() % TableSize];
                    if (target == indexOfContextWord)
                    {
                        // do not negative sample the context word
                        continue;
                    }
                    isPositiveSample = false;
                }
                neuralNetwork.NegativeSample(indexOfCurrentWord, target, _alpha, isPositiveSample);
            }
        }
    }
}