using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using GingerbreadAI.DeepLearning.Backpropagation;
using GingerbreadAI.DeepLearning.Backpropagation.ErrorFunctions;
using GingerbreadAI.DeepLearning.Backpropagation.Extensions;
using GingerbreadAI.Model.NeuralNetwork.ActivationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Extensions;
using GingerbreadAI.Model.NeuralNetwork.InitialisationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Models;
using GingerbreadAI.NLP.Word2Vec.WordCollectionExtensions;

namespace GingerbreadAI.NLP.Word2Vec
{
    public class Word2VecUsingLibrary
    {
        private const int MaxCodeLength = 40;
        private const int TableSize = (int)1e8;

        private readonly FileHandler _fileHandler;
        private readonly int _numberOfThreads;
        private readonly int _numberOfIterations;
        private readonly int _numberOfDimensions;
        private readonly int _maxSentenceLength;
        private readonly int _minCount;
        private readonly double _startingLearningRate;
        private readonly bool _useSkipgram;
        private readonly bool _useCbow;
        private readonly int _negativeSamples;
        private readonly int _windowSize;
        private readonly float _thresholdForOccurrenceOfWords;

        private double _learningRate;
        private int[] _table;
        private int _wordCountActual;
        private WordCollection _wordCollection;
        private Layer _neuralNetwork;

        public Word2VecUsingLibrary(
            FileHandler fileHandler,
            int numberOfThreads = 4,
            int numberOfIterations = 4,
            int numberOfDimensions = 50,
            int maxSentenceLength = 10000,
            int minCount = 5,
            float startingLearningRate = 0.025f,
            bool useSkipgram = true,
            bool useCbow = true,
            int negativeSamples = 5,
            int windowSize = 5,
            float thresholdForOccurrenceOfWords = 1e-3f
        )
        {
            _fileHandler = fileHandler;
            _numberOfThreads = numberOfThreads;
            _numberOfIterations = numberOfIterations;
            _numberOfDimensions = numberOfDimensions;
            _maxSentenceLength = maxSentenceLength;
            _minCount = minCount;
            _startingLearningRate = startingLearningRate;
            _useSkipgram = useSkipgram;
            _useCbow = useCbow;
            // note: first 'negative sample' is positive
            _negativeSamples = negativeSamples;
            _windowSize = windowSize;
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
            _learningRate = _startingLearningRate;

            _wordCollection = _fileHandler.GetWordDictionaryFromFile(MaxCodeLength);
            _wordCollection.RemoveWordsWithCountLessThanMinCount(_minCount);
            _wordCollection.CreateBinaryTree();
            if (_negativeSamples > 0)
            {
                _table = _wordCollection.GetUnigramTable(TableSize);
            }

            var inputLayer = new Layer(_wordCollection.GetNumberOfUniqueWords(), new Layer[0], ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var hiddenLayer = new Layer(_numberOfDimensions, new[] { inputLayer }, ActivationFunctionType.RELU, InitialisationFunctionType.HeUniform);
            _neuralNetwork = new Layer(_wordCollection.GetNumberOfUniqueWords(), new[] { hiddenLayer }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.HeEtAl);
            _neuralNetwork.AddMomentumRecursively();
            _neuralNetwork.Initialise(new Random());

            GC.Collect();
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

        private void TrainModelThreadStart(int id)
        {
            var sentenceLength = 0;
            var sentencePosition = 0;
            var wordCount = 0;
            var lastWordCount = 0;
            var sentence = new int?[_maxSentenceLength];
            var localIterations = _numberOfIterations;
            var localNetwork = _neuralNetwork.CloneWithSameWeightValueReferences();

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
                        _learningRate = _startingLearningRate * (1 - _wordCountActual / (float)(_numberOfIterations * sum + 1));
                        if (_learningRate < _startingLearningRate * (float)0.0001)
                        {
                            _learningRate = _startingLearningRate * (float)0.0001;
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
                        localIterations--;
                        if (localIterations == 0)
                        {
                            break;
                        }
                        wordCount = 0;
                        lastWordCount = 0;
                        sentenceLength = 0;
                        reader.BaseStream.Seek(_fileHandler.FileSize / _numberOfThreads * id, SeekOrigin.Begin);
                        Console.WriteLine($"Iterations remaining: {localIterations} Thread: {id}");
                        continue;
                    }
                    var wordIndex = sentence[sentencePosition];
                    if (!wordIndex.HasValue)
                    {
                        continue;
                    }

                    if (_negativeSamples > 0)
                    {
                        TrainNetwork(localNetwork, _table, sentence, sentencePosition, sentenceLength, wordIndex.Value,
                            _windowSize, _useSkipgram, _useCbow, _negativeSamples, _learningRate, random);
                    }

                    sentencePosition++;

                    if (sentencePosition >= sentenceLength)
                    {
                        sentenceLength = 0;
                    }
                }
            }
            GC.Collect();
        }

        private int SetSentence(StreamReader reader, int wordCount, int?[] sentence, Random random, ref int sentenceLength,
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

        private static void TrainNetwork(Layer neuralNetwork, int[] table, int?[] sentence, int sentencePosition, int sentenceLength, int indexOfCurrentWord, int windowSize,
            bool useSkipgram, bool useCbow, int negativeSamples, double learningRate, Random random)
        {
            var randomWindowPosition = (int)(random.Next() % (uint)windowSize);

            for (var offsetWithinWindow = randomWindowPosition; offsetWithinWindow < windowSize * 2 + 1 - randomWindowPosition; offsetWithinWindow++)
            {
                if (offsetWithinWindow != windowSize)
                {
                    var indexOfCurrentContextWordInSentence = sentencePosition - windowSize + offsetWithinWindow;

                    if (indexOfCurrentContextWordInSentence >= 0 && indexOfCurrentContextWordInSentence < sentenceLength)
                    {
                        var indexOfContextWord = sentence[indexOfCurrentContextWordInSentence];

                        if (indexOfContextWord.HasValue)
                        {
                            NegativeSampling(neuralNetwork, table, indexOfCurrentWord, indexOfContextWord.Value, 
                                useSkipgram, useCbow, negativeSamples, learningRate, random);
                        }
                    }
                }
            }
        }

        private static void NegativeSampling(Layer neuralNetwork, int[] table, int indexOfCurrentWord, int indexOfContextWord,
            bool useSkipgram, bool useCbow, int negativeSamples, double learningRate, Random random)
        {
            for (var i = 0; i < negativeSamples; i++)
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
                    target = table[random.Next() % TableSize];
                    if (target == indexOfContextWord)
                    {
                        // do not negative sample the context word
                        continue;
                    }
                    isPositiveSample = false;
                }

                if (useSkipgram)
                {
                    // current -> context
                    neuralNetwork.NegativeSample(indexOfCurrentWord, target, isPositiveSample, ErrorFunctionType.CrossEntropy, 0.01, 0.9);
                }
                if (useCbow)
                {
                    // context -> current
                    neuralNetwork.NegativeSample(target, indexOfCurrentWord, isPositiveSample, ErrorFunctionType.CrossEntropy, 0.01, 0.9);
                }
            }
        }
    }
}