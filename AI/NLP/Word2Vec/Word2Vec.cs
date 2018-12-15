using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

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
        private const int TableSize = (int) 1e8;
        private readonly float[] _expTable;
        private readonly long _numberOfIterations;
        private readonly int _numberOfDimensions;
        private readonly int _minCount = 5;
        private readonly int _negative;
        private readonly int _numberOfThreads = 4;
        private readonly string _outputFile;
        private readonly string _readVocabFile;
        private readonly float _thresholdForOccurrenceOfWords = 1e-3f;
        private readonly string _saveVocabFile;
        private readonly WordCollection _wordCollection = new WordCollection();
        private readonly string _trainFile;
        private readonly int _windowSize;
        private float _alpha;
        private long _fileSize;
        private float _startingAlpha;
        private float[,] _hiddenLayerWeights;
        private float[,] _outputLayerWeights;
        private int[] _table;
        
        private long _wordCountActual;

        public Word2Vec(
            string trainFileName,
            string outPutfileName,
            string saveVocabFileName,
            int numberOfThreads = 4,
            int numberOfIterations = 4,
            int negative = 5,
            int numberOfDimensions = 50,
            int windowSize = 5,
            float alpha = 0.025f
        )
        {
            _trainFile = trainFileName;
            _outputFile = outPutfileName;
            _saveVocabFile = saveVocabFileName;
            _expTable = new float[ExpTableSize + 1];
            _numberOfThreads = numberOfThreads;
            _numberOfIterations = numberOfIterations;
            _negative = negative;
            _numberOfDimensions = numberOfDimensions;
            _windowSize = windowSize;
            _alpha = alpha;
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
            WriteOutput();
            GC.Collect();
        }

        private void Setup()
        {
            _startingAlpha = _alpha;
            _fileSize = new FileInfo(_trainFile).Length;

            FileHandler.GetWordDictionaryFromFile(_trainFile, _wordCollection, MaxCodeLength);

            _wordCollection.RemoveWordsWithCountLessThanMinCount(_minCount);

            if (!string.IsNullOrEmpty(_saveVocabFile))
                FileHandler.SaveWordDictionary(_saveVocabFile, _wordCollection);

            if (string.IsNullOrEmpty(_outputFile))
                throw new Exception("Output file not defined.");

            InitNetwork();

            if (_negative > 0)
                InitUnigramTable();
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


        private void WriteOutput()
        {
            using (var fs = new FileStream(_outputFile, FileMode.Create, FileAccess.Write))
            using (var writer = new StreamWriter(fs, Encoding.UTF8))
            {
                writer.WriteLine(_wordCollection.GetNumberOfUniqueWords());
                writer.WriteLine(_numberOfDimensions);

                var keys = _wordCollection.GetWords().ToArray();
                for (var a = 0; a < _wordCollection.GetNumberOfUniqueWords(); a++)
                {
                    var bytes = new List<byte>();
                    for (var dimensionIndex = 0; dimensionIndex < _numberOfDimensions; dimensionIndex++)
                        bytes.AddRange(BitConverter.GetBytes(_hiddenLayerWeights[a, dimensionIndex]));
                    writer.WriteLine($"{keys[a]}\t{Convert.ToBase64String(bytes.ToArray())}");
                }
            }
        }

        private void InitNetwork()
        {
            long wordIndex, dimensionIndex;
            ulong nextRandom = 1;

            var numberOfWords = _wordCollection.GetNumberOfUniqueWords();
            _hiddenLayerWeights = new float[numberOfWords, _numberOfDimensions];
            if (_negative > 0)
            {
                _outputLayerWeights = new float[_wordCollection.GetNumberOfUniqueWords(), _numberOfDimensions];
                for (wordIndex = 0; wordIndex < _wordCollection.GetNumberOfUniqueWords(); wordIndex++)
                for (dimensionIndex = 0; dimensionIndex < _numberOfDimensions; dimensionIndex++)
                    _outputLayerWeights[wordIndex, dimensionIndex] = 0;
            }
            for (wordIndex = 0; wordIndex < _wordCollection.GetNumberOfUniqueWords(); wordIndex++)
                for (dimensionIndex = 0; dimensionIndex < _numberOfDimensions; dimensionIndex++)
                {
                    nextRandom = LinearCongruentialGenerator(nextRandom);
                    _hiddenLayerWeights[wordIndex, dimensionIndex] = ((nextRandom & 0xFFFF) / (float) 65536 - (float) 0.5) / _numberOfDimensions;
                }
            HuffmanTree.Create(_wordCollection, MaxCodeLength);
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
            var d1 = Math.Pow(_wordCollection.GetOccuranceOfWord(keys.First()), power) / trainWordsPow;
            for (a = 0; a < TableSize; a++)
            {
                _table[a] = i;
                if (a / (double) TableSize > d1)
                {
                    i++;
                    d1 += Math.Pow(_wordCollection.GetOccuranceOfWord(keys[i]), power) / trainWordsPow;
                }
                if (i >= _wordCollection.GetNumberOfUniqueWords())
                    i = _wordCollection.GetNumberOfUniqueWords() - 1;
            }
        }

        private void TrainModelThreadStart(int id)
        {
            var splitRegex = new Regex("\\s");
            long sentenceLength = 0;
            long sentencePosition = 0;
            long wordCount = 0, lastWordCount = 0;
            var sentence = new long?[MaxSentenceLength + 1]; //Sentence elements will not be null to my understanding
            var localIter = _numberOfIterations;

            var nextRandom = (ulong) id;
            var neu1 = new float[_numberOfDimensions];
            var sum = _wordCollection.GetTotalNumberOfWords();
            string[] lastLine = null;
            using (var reader = File.OpenText(_trainFile))
            {
                reader.BaseStream.Seek(_fileSize / _numberOfThreads * id, SeekOrigin.Begin);
                while (true)
                {
                    if (wordCount - lastWordCount > 10000)
                    {
                        _wordCountActual += wordCount - lastWordCount;
                        lastWordCount = wordCount;
                        _alpha = _startingAlpha * (1 - _wordCountActual / (float) (_numberOfIterations * sum + 1));
                        if (_alpha < _startingAlpha * (float) 0.0001)
                            _alpha = _startingAlpha * (float) 0.0001;
                    }
                    if (sentenceLength == 0)
                    {
                        wordCount = SetSentence(reader, splitRegex, wordCount, sum, sentence, ref nextRandom, ref sentenceLength, ref lastLine);
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
                        reader.BaseStream.Seek(_fileSize / _numberOfThreads * id, SeekOrigin.Begin);
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
                    var randomWindowPosition = (long) (nextRandom % (ulong) _windowSize);
                    
                    nextRandom = SkipGram(randomWindowPosition, sentencePosition, sentenceLength, sentence, wordIndex.Value, nextRandom);
                    sentencePosition++;
                    if (sentencePosition >= sentenceLength)
                        sentenceLength = 0;
                }
            }
            GC.Collect();
        }

        private long SetSentence(StreamReader reader, Regex splitRegex, long wordCount, long sum, long?[] sentence,
            ref ulong nextRandom, ref long sentenceLength, ref string [] lastLine)
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

        private bool HandleWords(StreamReader reader, ref long wordCount, long sum, long?[] sentence, ref ulong nextRandom,
            ref long sentenceLength, IEnumerable<string> words)
        {
            var loopEnd = false;
            foreach (var word in words)
            {
                var wordIndex = _wordCollection[word];
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
                    var ran = ((float) Math.Sqrt(_wordCollection.GetOccuranceOfWord(word) /
                                                 (_thresholdForOccurrenceOfWords * sum)) + 1) *
                              (_thresholdForOccurrenceOfWords * sum) / _wordCollection.GetOccuranceOfWord(word);
                    nextRandom = LinearCongruentialGenerator(nextRandom);
                    if (ran < (nextRandom & 0xFFFF) / (float) 65536)
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
        private ulong SkipGram(long randomWindowPosition, long sentencePosition, long sentenceLength, long?[] sentence, long word,
            ulong nextRandom)
        {
            // Loop over the positions in the context window (skipping the word at
            // the center). 'a' is just the offset within the window, it's not 
            // the index relative to the beginning of the sentence.
            //train skip-gram
            for (var offsetWithinWindow = randomWindowPosition; offsetWithinWindow < _windowSize * 2 + 1 - randomWindowPosition; offsetWithinWindow++)
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
                        continue;
                    
                    // NEGATIVE SAMPLING
                    if (_negative > 0)
                        nextRandom = NegativeSampling(word, nextRandom, indexOfContextWord.Value,
                            _negative, _table, _wordCollection.GetNumberOfUniqueWords(), _numberOfDimensions, _hiddenLayerWeights,
                            _outputLayerWeights, _alpha, _expTable);
                }
            return nextRandom;
        }

        private static float[] InitOutputErrorContainer(int numberOfDimensions)
        {
            var accumulatedOutputError = new float[numberOfDimensions];
            for (var dimensionIndex = 0; dimensionIndex < numberOfDimensions; dimensionIndex++)
                accumulatedOutputError[dimensionIndex] = 0;
            return accumulatedOutputError;
        }

        private static ulong NegativeSampling(long currentWordIndex, ulong nextRandom,
            long indexOfContextWord, int numberOfNegativeSamples, int[] table,
            long numberOfWords, int numberOfDimensions, float[,] hiddenLayerWeights,
            float[,] outputLayerWeights,
            float trainingRate, float[] expTable)
        {
            var accumulatedOutputError = InitOutputErrorContainer(numberOfDimensions);

            // NEGATIVE SAMPLING
            // Rather than performing backpropagation for every word in our 
            // vocabulary, we only perform it for a few words (the number of words is given by 'negative').
            // These words are selected using a "unigram" distribution, which is generated in the function InitUnigramTable
            for (var index = 0; index <= numberOfNegativeSamples; index++)
            {
                long label;
                long target;
                if (index == 0)
                {
                    // On the first iteration, we're going to train the positive sample.
                    target = currentWordIndex;
                    label = 1;
                }
                else
                {
                    target = SelectTarget(ref nextRandom, table, numberOfWords);
                    // Don't use the positive sample as a negative sample!
                    if (target == currentWordIndex)
                        continue;
                    // Mark this as a negative example.
                    label = 0;
                }
                // Get the index of the target word in the output layer.

                // At this point, our two words are represented by their index into the layer weights.
                // l1 - The index of our input word within the hidden layer weights.
                // l2 - The index of our output word within the output layer weights.
                // label - Whether this is a positive (1) or negative (0) example.

                float dotProduct = 0;
                // Calculate the dot-product between the input words weights (in 
                // syn0) and the output word's weights (in syn1neg).
                for (var dimensionIndex = 0; dimensionIndex < numberOfDimensions; dimensionIndex++)
                    dotProduct += hiddenLayerWeights[indexOfContextWord, dimensionIndex] * outputLayerWeights[target, dimensionIndex];
                float outputError;
                // This block does two things:
                //   1. Calculates the output of the network for this training
                //      pair, using the expTable to evaluate the output layer
                //      activation function.
                //   2. Calculate the error at the output, stored in 'g', by
                //      subtracting the network output from the desired output, 
                //      and finally multiply this by the learning rate.
                if (dotProduct > MaxExp) //DotProduct > MaxExp. G will be negative for negative sample.
                    outputError = (label - 1) * trainingRate;
                else if (dotProduct < MaxExp * -1) // DotProduct < -MaxExp. G will be Zero for negative sample. 
                    outputError = (label - 0) * trainingRate;
                else //-MaxExp < dotproduct < MaxExp. 
                    outputError = (label - expTable[(int) ((dotProduct + MaxExp) * (ExpTableSize / (float) MaxExp / 2))]) * trainingRate;

                for (var dimensionIndex = 0; dimensionIndex < numberOfDimensions; dimensionIndex++)
                {
                    // Multiply the error by the output layer weights.
                    // (I think this is the gradient calculation?)
                    // Accumulate these gradients over all of the negative samples.
                    accumulatedOutputError[dimensionIndex] += outputError * outputLayerWeights[target, dimensionIndex];

                    // Update the output layer weights by multiplying the output error
                    // by the hidden layer weights.
                    outputLayerWeights[target, dimensionIndex] += outputError * hiddenLayerWeights[indexOfContextWord, dimensionIndex];
                }
            }

            // Learn weights input -> hidden
            // Once the hidden layer gradients for all of the negative samples have
            // been accumulated, update the hidden layer weights.
            for (var dimensionIndex = 0; dimensionIndex < numberOfDimensions; dimensionIndex++)
                hiddenLayerWeights[indexOfContextWord, dimensionIndex] += accumulatedOutputError[dimensionIndex];
            return nextRandom;
        }

        private static long SelectTarget(ref ulong nextRandom, int[] table, long numberOfWords)
        {
            nextRandom = LinearCongruentialGenerator(nextRandom);
            // 'target' becomes the index of the word in the vocab to use as the negative sample.
            long target = table[(nextRandom >> 16) % TableSize];
            // If the target is the special end of sentence token, then just pick a random word from the vocabulary instead.
            if (target == 0)
                target = (long) (nextRandom % (ulong) (numberOfWords - 1) + 1);
            return target;
        }
    }
}