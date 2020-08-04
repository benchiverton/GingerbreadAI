using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GingerbreadAI.DeepLearning.Backpropagation;
using GingerbreadAI.DeepLearning.Backpropagation.ErrorFunctions;
using GingerbreadAI.Model.NeuralNetwork.ActivationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Extensions;
using GingerbreadAI.Model.NeuralNetwork.InitialisationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Models;
using GingerbreadAI.NLP.Word2Vec.Embeddings;
using GingerbreadAI.NLP.Word2Vec.Extensions;

namespace GingerbreadAI.NLP.Word2Vec
{
    public class Word2VecTrainer
    {
        private const int MaxCodeLength = 40;
        private const int TableSize = (int)1e8;

        private FileHandler _trainingFileHandler;
        private int _wordCountActual;
        private int[] _table;

        public WordCollection WordCollection { get; private set; }
        public Layer NeuralNetwork { get; private set; }

        public void Setup(string trainingFileLocation, int dimensions = 50, int minWordOccurrences = 5)
        {
            _trainingFileHandler = new FileHandler(trainingFileLocation);

            WordCollection = _trainingFileHandler.GetWordDictionaryFromFile(MaxCodeLength);
            WordCollection.RemoveWordsWithCountLessThanMinCount(minWordOccurrences);
            _table = WordCollection.GetUnigramTable(TableSize);

            var inputLayer = new Layer(WordCollection.GetNumberOfUniqueWords(), Array.Empty<Layer>(), ActivationFunctionType.RELU, InitialisationFunctionType.None);
            var hiddenLayer = new Layer(dimensions, new[] { inputLayer }, ActivationFunctionType.Linear, InitialisationFunctionType.RandomWeighted, false);
            NeuralNetwork = new Layer(WordCollection.GetNumberOfUniqueWords(), new[] { hiddenLayer }, ActivationFunctionType.Sigmoid, InitialisationFunctionType.HeEtAl, false);
            NeuralNetwork.Initialise(new Random());

            GC.Collect();
        }

        public void TrainModel(
            int numberOfThreads = 4,
            int numberOfIterations = 4,
            int maxSentenceLength = 10000,
            double startingLearningRate = 0.025,
            bool useSkipgram = true,
            bool useCbow = true,
            int negativeSamples = 5,
            int windowSize = 5,
            double thresholdForOccurrenceOfWords = 1e-3)
        {
            if (NeuralNetwork == null)
            {
                throw new Exception("The network has not been configured. Please call the 'Setup' method.");
            }

            var learningRate = startingLearningRate;

            var parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = numberOfThreads
            };
            WordCollection.InitWordPositions();
            var result = Parallel.For(0, numberOfThreads, parallelOptions, i => TrainModelThreadStart(
                i,
                numberOfThreads,
                numberOfIterations,
                maxSentenceLength,
                startingLearningRate,
                ref learningRate,
                useSkipgram,
                useCbow,
                negativeSamples,
                windowSize,
                thresholdForOccurrenceOfWords));
            if (!result.IsCompleted)
            {
                throw new InvalidOperationException();
            }

            GC.Collect();
        }

        public void WriteWordEmbeddings(string embeddingsFileLocation)
        {
            var words = WordCollection.GetWords().ToArray();
            var hiddenLayer = NeuralNetwork.PreviousLayers[0];
            var inputLayer = hiddenLayer.PreviousLayers[0];

            var wordEmbeddings = new List<WordEmbedding>();
            for (var i = 0; i < WordCollection.GetNumberOfUniqueWords(); i++)
            {
                wordEmbeddings.Add(new WordEmbedding(words[i], hiddenLayer.Nodes.Select(hiddenNode => hiddenNode.Weights[inputLayer.Nodes[i]].Value).ToArray()));
            }
            using var fileStream = new FileStream(embeddingsFileLocation, FileMode.OpenOrCreate, FileAccess.Write);
            _ = fileStream.Seek(0, SeekOrigin.End);
            using var writer = new StreamWriter(fileStream, Encoding.UTF8);
            wordEmbeddings.WriteEmbeddingToStream(writer);
        }

        private void TrainModelThreadStart(
            int id,
            int numberOfThreads,
            int numberOfIterations,
            int maxSentenceLength,
            double startingLearningRate,
            ref double learningRate,
            bool useSkipgram,
            bool useCbow,
            int negativeSamples,
            int windowSize,
            double thresholdForOccurrenceOfWords)
        {
            var sentenceLength = 0;
            var sentencePosition = 0;
            var wordCount = 0;
            var lastWordCount = 0;
            var sentence = new int?[maxSentenceLength];
            var localIterations = numberOfIterations;
            var localNetwork = NeuralNetwork.CloneWithDifferentOutputs();

            var random = new Random();
            var totalNumberOfWords = WordCollection.GetTotalNumberOfWords();
            string[] lastLine = null;
            using (var reader = _trainingFileHandler.GetTrainingFileReader())
            {
                reader.BaseStream.Seek(_trainingFileHandler.FileSize / numberOfThreads * id, SeekOrigin.Begin);
                while (true)
                {
                    // adjust learning rate
                    if (wordCount - lastWordCount > 10000)
                    {
                        _wordCountActual += wordCount - lastWordCount;
                        lastWordCount = wordCount;
                        learningRate = startingLearningRate * (1 - (_wordCountActual / (float)((numberOfIterations * totalNumberOfWords) + 1)));
                        if (learningRate < startingLearningRate * (float)0.0001)
                        {
                            learningRate = startingLearningRate * (float)0.0001;
                        }
                    }
                    // if sentence length is 0, set the sentence
                    if (sentenceLength == 0)
                    {
                        SetSentence(WordCollection, reader, sentence, random, ref sentenceLength, ref lastLine, thresholdForOccurrenceOfWords);
                        sentencePosition = 0;
                    }
                    // if sentence length is still 0, no sentence could be set (last iteration was the end of the stream)
                    // if wordCount > sum / numberOfThreads, we are 'leaking' into another threads work
                    if (sentenceLength == 0 || wordCount > totalNumberOfWords / numberOfThreads)
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
                        reader.BaseStream.Seek(_trainingFileHandler.FileSize / numberOfThreads * id, SeekOrigin.Begin);
                        Console.WriteLine($"Iterations remaining: {localIterations} Thread: {id}");
                        continue;
                    }
                    // word may not be populated (as frequent words are sub-sampled)
                    var wordIndex = sentence[sentencePosition];
                    if (!wordIndex.HasValue)
                    {
                        continue;
                    }

                    TrainModelFromContextWindow(
                        localNetwork,
                        _table,
                        sentence,
                        sentencePosition,
                        sentenceLength,
                        wordIndex.Value,
                        windowSize,
                        useSkipgram,
                        useCbow,
                        negativeSamples,
                        learningRate,
                        random);

                    sentencePosition++;
                    wordCount++;

                    if (sentencePosition >= sentenceLength)
                    {
                        sentenceLength = 0;
                    }
                }
            }
            GC.Collect();
        }

        internal static void SetSentence(
            WordCollection wordCollection,
            StreamReader reader,
            int?[] sentence,
            Random random,
            ref int sentenceLength,
            ref string[] lineThatGotCutOff,
            double thresholdForOccurrenceOfWords)
        {
            string line;
            var loopEnd = false;
            if (lineThatGotCutOff != null && lineThatGotCutOff.Any())
            {
                loopEnd = HandleWords(
                    wordCollection,
                    reader,
                    sentence,
                    random,
                    ref sentenceLength,
                    lineThatGotCutOff,
                    thresholdForOccurrenceOfWords);
                lineThatGotCutOff = null;
            }

            while (!loopEnd && (line = reader.ReadLine()) != null)
            {
                var words = WordCollection.ParseWords(line).ToArray();
                if (words.Length > sentence.Length)
                {
                    continue;
                }
                if (sentenceLength > sentence.Length - words.Length)
                {
                    lineThatGotCutOff = words;
                    break;
                }
                loopEnd = HandleWords(
                    wordCollection,
                    reader,
                    sentence,
                    random,
                    ref sentenceLength,
                    words,
                    thresholdForOccurrenceOfWords);
            }
        }

        private static bool HandleWords(
            WordCollection wordCollection,
            StreamReader reader,
            int?[] sentence,
            Random random,
            ref int sentenceLength,
            IEnumerable<string> words,
            double thresholdForOccurrenceOfWords)
        {
            var totalNumberOfWords = wordCollection.GetTotalNumberOfWords();
            foreach (var word in words)
            {
                var wordIndex = wordCollection[word];
                if (!wordIndex.HasValue)
                {
                    continue;
                }

                //Sub-sampling of frequent words
                if (thresholdForOccurrenceOfWords > 0)
                {
                    var something = ((float)Math.Sqrt(wordCollection.GetOccurrenceOfWord(word) / (thresholdForOccurrenceOfWords * totalNumberOfWords)) + 1)
                                    * (thresholdForOccurrenceOfWords * totalNumberOfWords) / wordCollection.GetOccurrenceOfWord(word);
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

        /// <summary>
        /// Iterates through the context window and trains the model from each context word.
        /// </summary>
        private static void TrainModelFromContextWindow(
            Layer neuralNetwork,
            int[] table,
            int?[] sentence,
            int sentencePosition,
            int sentenceLength,
            int indexOfCurrentWord,
            int windowSize,
            bool useSkipgram,
            bool useCbow,
            int negativeSamples,
            double learningRate,
            Random random)
        {
            var randomWindowPosition = (int)(random.Next() % (uint)windowSize);

            for (var offsetWithinWindow = randomWindowPosition; offsetWithinWindow < (windowSize * 2) + 1 - randomWindowPosition; offsetWithinWindow++)
            {
                if (offsetWithinWindow != windowSize)
                {
                    var indexOfCurrentContextWordInSentence = sentencePosition - windowSize + offsetWithinWindow;

                    if (indexOfCurrentContextWordInSentence >= 0 && indexOfCurrentContextWordInSentence < sentenceLength)
                    {
                        var indexOfContextWord = sentence[indexOfCurrentContextWordInSentence];

                        if (indexOfContextWord.HasValue)
                        {
                            TrainModelFromContextWord(
                                neuralNetwork,
                                table,
                                indexOfCurrentWord,
                                indexOfContextWord.Value,
                                useSkipgram,
                                useCbow,
                                negativeSamples,
                                learningRate,
                                random);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Trains the model from a context word.
        /// Positively samples the context word.
        /// Negatively samples 'negativeSamples - 1' random words (that aren't the context word).
        /// </summary>
        private static void TrainModelFromContextWord(
            Layer neuralNetwork,
            int[] table,
            int indexOfCurrentWord,
            int indexOfContextWord,
            bool useSkipgram,
            bool useCbow,
            int negativeSamples,
            double learningRate, Random random)
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
                    neuralNetwork.NegativeSample(
                        indexOfCurrentWord,
                        target,
                        isPositiveSample,
                        ErrorFunctionType.CrossEntropy,
                        learningRate);
                }
                if (useCbow)
                {
                    // context -> current
                    neuralNetwork.NegativeSample(
                        target,
                        indexOfCurrentWord,
                        isPositiveSample,
                        ErrorFunctionType.CrossEntropy,
                        learningRate);
                }
            }
        }
    }
}
