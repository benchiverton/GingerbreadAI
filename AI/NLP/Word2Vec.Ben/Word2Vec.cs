namespace Word2Vec.Ben
{
    using NegativeSampling;
    using NeuralNetwork;
    using NeuralNetwork.Data;
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Threading.Tasks;

    //C# Word2Vec based on https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c
    //Any comments in here is to show how misleading they can be. :)
    public class Word2Vec
    {
        private const int ExpTableSize = 1000;
        private const int MaxCodeLength = 40;
        private const double MaxExp = 0.778; // Log(6)
        private readonly int _maxSentenceLength;
        private readonly long _numberOfIterations;
        private readonly int _numberOfDimensions;
        private readonly int _minCount = 5;
        private readonly int _negative;
        private readonly int _numberOfThreads = 4;
        private readonly float _thresholdForOccurrenceOfWords = 1e-3f;
        private readonly WordCollection _wordCollection = new WordCollection();
        private readonly int _windowSize;
        private readonly FileHandler _fileHandler;
        private int[] _table;
        private const int TableSize = (int)1e8;

        private Layer _network;
        private Func<double, double> _learningRateModifier;

        private long _wordCountActual;
        private readonly bool _useHs;

        public Word2Vec(
            string trainFileName,
            string outputFileName,
            int numberOfThreads = 4,
            int numberOfIterations = 4,
            int negative = 5,
            int numberOfDimensions = 50,
            int windowSize = 5,
            float alpha = 0.025f,
            bool useHs = false,
            int maxSentenceLength = 10000
        )
        {
            _numberOfThreads = numberOfThreads;
            _numberOfIterations = numberOfIterations;
            _negative = negative;
            _numberOfDimensions = numberOfDimensions;
            _windowSize = windowSize;
            _fileHandler = new FileHandler(trainFileName, outputFileName);
            _useHs = useHs;
            _maxSentenceLength = maxSentenceLength;
        }

        public void TrainModel()
        {
            Setup();
            Train();
            _fileHandler.WriteOutput(_wordCollection, _network);
            GC.Collect();
        }

        private void Setup()
        {
            _fileHandler.GetWordDictionaryFromFile(_wordCollection, MaxCodeLength);

            _wordCollection.RemoveWordsWithCountLessThanMinCount(_minCount);

            Initialise();

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
                throw new InvalidOperationException();
        }


        private void Initialise()
        {
            var numberOfWords = _wordCollection.GetNumberOfUniqueWords();

            var input = new Layer("input", numberOfWords, new Layer[0]);
            var hidden = new Layer("hidden1", _numberOfDimensions, new[] { input });
            var output = new Layer("output", numberOfWords, new[] { hidden });
            LayerInitialiser.Initialise(new Random(), output);

            _network = output;

            _learningRateModifier = (rate) => rate * 0.99 < 0.1 ? 0.1 : rate * 0.99;

            var huffmanTree = new HuffmanTree();
            huffmanTree.Create(_wordCollection);
            GC.Collect();
        }

        private static ulong LinearCongruentialGenerator(ulong nextRandom)
        {
            return nextRandom * 25214903917 + 11;
        }

        private void InitUnigramTable()
        {
            if (_wordCollection.GetNumberOfUniqueWords() == 0)
                return;

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
                    i = _wordCollection.GetNumberOfUniqueWords() - 1;
            }
        }

        private void TrainModelThreadStart(int id)
        {
            var network = _network.CloneNewWithWeightReferences();
            var negativeSampler = new NegativeSampler(network, 0.025, _learningRateModifier);

            long sentenceLength = 0;
            long sentencePosition = 0;
            long wordCount = 0, lastWordCount = 0;
            var sentence = new long?[_maxSentenceLength]; //Sentence elements will not be null to my understanding
            var localIter = _numberOfIterations;

            var nextRandom = (ulong)id;
            var neu1 = new float[_numberOfDimensions];
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
                    }
                    if (sentenceLength == 0)
                    {
                        wordCount = SetSentence(reader, wordCount, sentence, ref nextRandom, ref sentenceLength, ref lastLine, _wordCollection, _thresholdForOccurrenceOfWords);
                        sentencePosition = 0;
                    }
                    if (reader.EndOfStream || wordCount > sum / _numberOfThreads)
                    {
                        _wordCountActual += wordCount - lastWordCount;
                        localIter--;
                        if (localIter == 0)
                            break;
                        wordCount = 0;
                        lastWordCount = 0;
                        sentenceLength = 0;
                        reader.BaseStream.Seek(_fileHandler.FileSize / _numberOfThreads * id, SeekOrigin.Begin);
                        Console.WriteLine($"Iterations remaining: {localIter} Thread: {id}");
                        continue;
                    }
                    var wordIndex = sentence[sentencePosition];
                    if (!wordIndex.HasValue)
                        continue;
                    long c;
                    for (c = 0; c < _numberOfDimensions; c++)
                        neu1[c] = 0;
                    nextRandom = LinearCongruentialGenerator(nextRandom);
                    var randomWindowPosition = (long)(nextRandom % (ulong)_windowSize);

                    nextRandom = SkipGram(randomWindowPosition, sentencePosition, sentenceLength, sentence, wordIndex.Value, nextRandom, network, negativeSampler);
                    sentencePosition++;
                    if (sentencePosition >= sentenceLength)
                    {
                        sentenceLength = 0;
                    }
                }
            }
            GC.Collect();
        }

        public static long SetSentence(StreamReader reader, long wordCount, long?[] sentence,
            ref ulong nextRandom, ref long sentenceLength, ref string[] lineThatGotCutOff, WordCollection wordCollection, float thresholdForOccurrenceOfWords)
        {
            string line;
            var loopEnd = false;
            if (lineThatGotCutOff != null && lineThatGotCutOff.Any())
            {
                loopEnd = HandleWords(reader, ref wordCount, sentence, ref nextRandom, ref sentenceLength, lineThatGotCutOff, wordCollection, thresholdForOccurrenceOfWords);
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
                loopEnd = HandleWords(reader, ref wordCount, sentence, ref nextRandom, ref sentenceLength, words, wordCollection, thresholdForOccurrenceOfWords);
            }
            return wordCount;
        }

        private static bool HandleWords(StreamReader reader, ref long wordCount, long?[] sentence, ref ulong nextRandom,
            ref long sentenceLength, IEnumerable<string> words, WordCollection wordCollection, float thresholdForOccurrenceOfWords)
        {
            var totalNumberOfWords = wordCollection.GetTotalNumberOfWords();
            foreach (var word in words)
            {
                var wordIndex = wordCollection[word];
                if (!wordIndex.HasValue)
                    continue;
                wordCount++;

                //Subsampling of frequent words
                if (thresholdForOccurrenceOfWords > 0)
                {
                    var random = ((float)Math.Sqrt(wordCollection.GetOccurrenceOfWord(word) / (thresholdForOccurrenceOfWords * totalNumberOfWords)) + 1) *
                              (thresholdForOccurrenceOfWords * totalNumberOfWords) / wordCollection.GetOccurrenceOfWord(word);
                    nextRandom = LinearCongruentialGenerator(nextRandom);
                    if (random < (nextRandom & 0xFFFF) / (float)65536)
                        continue;
                }
                sentence[sentenceLength] = wordIndex.Value;
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

        /* 
        * ====================================
        *        Skip-gram Architecture
        * ====================================
        * sen - This is the array of words in the sentence. Subsampling has already been
        *       applied. I don't know what the word representation is...
        *
        * sentence_position - This is the index of the current input word.
        *
        * a - Offset into the current window, relative to the window start.
        *     a will range from 0 to (window * 2) (TODO - not sure if it's inclusive or
        *      not).
        *
        * c - 'c' is the index of the current context word *within the sentence*
        *
        * syn0 - The hidden layer weights.
        *
        * l1 - Index into the hidden layer (syn0). Index of the start of the
        *      weights for the current input word.
        */


        //   'word' - The word at our current position in the sentence (in the center of a context window).
        private ulong SkipGram(long randomWindowPosition, long sentencePosition, long sentenceLength, long?[] sentence,
            long word, ulong nextRandom, Layer network = null, NegativeSampler negativeSampler = null)
        {
            // Loop over the positions in the context window (skipping the word at
            // the center). 'a' is just the offset within the window, it's not 
            // the index relative to the beginning of the sentence.
            //train skip-gram
            for (var offsetWithinWindow = randomWindowPosition; offsetWithinWindow < _windowSize * 2 + 1 - randomWindowPosition; offsetWithinWindow++)
            {
                if (offsetWithinWindow != _windowSize)
                {
                    // Convert the window offset 'a' into an index 'c' into the sentence 
                    // array.
                    var indexOfCurrentContextWordInSentence = sentencePosition - _windowSize + offsetWithinWindow;

                    // Verify c isn't outisde the bounds of the sentence.
                    if (indexOfCurrentContextWordInSentence < 0 || indexOfCurrentContextWordInSentence >= sentenceLength)
                        continue;

                    // Get the context word. That is, get the id of the word (its index in
                    // the vocab table).
                    var indexOfContextWord = sentence[indexOfCurrentContextWordInSentence];


                    // At this point we have two words identified:
                    //   'word' - The word at our current position in the sentence (in the
                    //            center of a context window).
                    //   'last_word' - The word at a position within the context window.

                    // Verify that the last word exists in the vocab (I don't think this should
                    // ever be the case?)
                    if (!indexOfContextWord.HasValue)
                    {
                        continue;
                    }

                    if (_useHs)
                    {
                        HierarchicalSoftmax(word, (int)indexOfContextWord, network, negativeSampler);
                    }
                    else if (_negative > 0)
                    {
                        nextRandom = NegativeSampling(word, nextRandom, indexOfContextWord.Value, _negative, _table, _wordCollection.GetNumberOfUniqueWords(), _numberOfDimensions, negativeSampler);
                    }
                }
            }
            return nextRandom;
        }

        private void HierarchicalSoftmax(long word, int indexOfContextWord, Layer network, NegativeSampler negativeSampler)
        {
            // TODO - this is a complete guess as to how this stuff works
            for (var d = 0; d < _wordCollection[word].CodeLength; d++)
            {
                var pointThing = (int)_wordCollection[word].Point[d];
                var result = network.GetResult(indexOfContextWord, pointThing);

                if (result <= MaxExp * -1) continue;
                if (result >= MaxExp) continue;

                negativeSampler.NegativeSample(indexOfContextWord, pointThing, true); // I think this might want to be true
            }
        }

        private float GetGradient(long word, int d, float dotProduct)
        {
            return (1 - _wordCollection[word].Code[d] - dotProduct);
        }

        private static ulong NegativeSampling(long currentWordIndex, ulong nextRandom,
            long indexOfContextWord, int numberOfNegativeSamples, int[] table,
            long numberOfWords, int numberOfDimensions, NegativeSampler negativeSampler)
        {
            // NEGATIVE SAMPLING
            // Rather than performing backpropagation for every word in our 
            // vocabulary, we only perform it for a few words (the number of words is given by 'negative').
            // These words are selected using a "unigram" distribution, which is generated in the function InitUnigramTable
            for (var index = 0; index <= numberOfNegativeSamples; index++)
            {
                bool isPositiveSample;
                long target;
                if (index == 0)
                {
                    // On the first iteration, we're going to train the positive sample.
                    target = currentWordIndex;
                    isPositiveSample = true;
                }
                else
                {
                    target = SelectTarget(ref nextRandom, table, numberOfWords);
                    // Don't use the positive sample as a negative sample!
                    if (target == currentWordIndex)
                        continue;
                    // Mark this as a negative example.
                    isPositiveSample = false;
                }

                negativeSampler.NegativeSample((int)currentWordIndex, (int)target, isPositiveSample);
            }

            return nextRandom;
        }

        private static long SelectTarget(ref ulong nextRandom, int[] table, long numberOfWords)
        {
            nextRandom = LinearCongruentialGenerator(nextRandom);
            // 'target' becomes the index of the word in the vocab to use as the negative sample.
            long target = table[(nextRandom >> 16) % TableSize];
            return target;
        }
    }
}