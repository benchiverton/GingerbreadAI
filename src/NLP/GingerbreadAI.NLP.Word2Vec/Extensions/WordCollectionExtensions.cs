using System;
using System.Collections.Generic;
using System.Linq;
using GingerbreadAI.Model.NeuralNetwork.Models;
using GingerbreadAI.NLP.Word2Vec.AnalysisFunctions;

namespace GingerbreadAI.NLP.Word2Vec.Extensions
{
    public static class WordCollectionExtensions
    {
        /// <summary>
        /// Create unigram table for random sub-sampling.
        /// Frequent words will occupy more positions in the table.
        /// </summary>
        public static int[] GetUnigramTable(this WordCollection wordCollection, int tableSize, double power = 0.75)
        {
            if (wordCollection.GetNumberOfUniqueWords() == 0)
            {
                // maybe throw?
                return new int[0];
            }

            var table = new int[tableSize];
            var sumOfOccurenceOfWordsRaisedToPower = wordCollection.GetSumOfOccurenceOfWordsRaisedToPower(power);

            var words = wordCollection.GetWords().ToArray();
            var indexOfCurrentWord = -1;
            var highestPositionOfWordInTable = -1;
            for (var tablePosition = 0; tablePosition < tableSize; tablePosition++)
            {
                if (tablePosition > highestPositionOfWordInTable)
                {
                    indexOfCurrentWord++;
                    highestPositionOfWordInTable += (int)Math.Ceiling(Math.Pow(wordCollection.GetOccurrenceOfWord(words[indexOfCurrentWord]), power) / sumOfOccurenceOfWordsRaisedToPower * tableSize);
                }

                table[tablePosition] = indexOfCurrentWord;

                if (indexOfCurrentWord >= wordCollection.GetNumberOfUniqueWords())
                {
                    indexOfCurrentWord = wordCollection.GetNumberOfUniqueWords() - 1;
                }
            }

            return table;
        }

        /// <summary>
        /// Returns each word in the word collection with their associated vector.
        /// </summary>
        public static IEnumerable<(string word, double[] vector)> GetWordVectors(this WordCollection wordCollection, Layer neuralNetwork)
        {
            var words = wordCollection.GetWords().ToArray();
            var hiddenLayer = neuralNetwork.PreviousLayers[0];
            var inputLayer = hiddenLayer.PreviousLayers[0];

            for (var i = 0; i < wordCollection.GetNumberOfUniqueWords(); i++)
            {
                yield return (words[i], hiddenLayer.Nodes.Select(hiddenNode => hiddenNode.Weights[inputLayer.Nodes[i]].Value).ToArray());
            }
        }

        /// <summary>
        /// Returns each word in the word collection with their associated vector.
        /// </summary>
        //public static IEnumerable<(string word, IEnumerable<(string word, double similarity)> similarWords)> GetMostSimilarWords(this WordCollection wordCollection, Layer neuralNetwork, int topn = 10)
        //{
        //    foreach (var word in wordCollection.GetWords())
        //    {
        //        yield return (word, WordVectorAnalysisFunctions.GetMostSimilarWords(word, wordCollection.GetWordVectors(neuralNetwork), topn));
        //    }
        //}

        /// <summary>
        /// Create binary Huffman tree using the word counts.
        /// Frequent words will have short unique binary codes.
        /// Huffman encoding is used for loss-less compression.
        /// The vocab_word structure contains a field for the 'code' for the word.
        /// </summary>
        public static void CreateBinaryTree(this WordCollection wordCollection)
        {
            var queue = GetQueueSortedByWordFrequencyAscending(wordCollection);
            IterateQueue(wordCollection, queue);
            var root = queue.Single();
            root.Code = "";
            Preorder(wordCollection, root);
            GC.Collect();
        }

        private static void IterateQueue(WordCollection wordCollection, List<Node> queue)
        {
            var numberOfInteriorNodes = 0;

            for (var a = 0; a < wordCollection.GetNumberOfUniqueWords() - 1; a++)
            {
                var node = CreateInteriorNode(queue, numberOfInteriorNodes);
                numberOfInteriorNodes++;
                InsertNodeInQueue(queue, node);
            }
        }

        private static Node CreateInteriorNode(List<Node> queue, int numberOfInteriorNodes)
        {
            var left = GetNodeFromQueue(queue);
            var right = GetNodeFromQueue(queue);
            var node = new Node
            {
                Left = left,
                Right = right,
                IndexOfLeafNodeThisInteriorNodePretendsToBe = numberOfInteriorNodes
            };
            node.Left.Parent = node;
            node.Right.Parent = node;
            node.Frequency = node.Left.Frequency + node.Right.Frequency;
            return node;
        }

        private static Node GetNodeFromQueue(List<Node> queue)
        {
            var node = queue.First();
            queue.Remove(node);
            return node;
        }

        private static void InsertNodeInQueue(List<Node> queue, Node node)
        {
            var index = queue.BinarySearch(node);
            if (index >= 0)
            {
                queue.Insert(index, node);
            }
            else
            {
                queue.Insert(~index, node);
            }
        }

        private static List<Node> GetQueueSortedByWordFrequencyAscending(WordCollection wordCollection)
        {
            var sortedByLowestCount = wordCollection.ToArray();
            var queue = sortedByLowestCount.Select(word => new Node
            {
                Frequency = word.Value.Count,
                WordInfo = word.Value,
                Word = word.Key
            })
                .OrderBy(y => y.Frequency).ToList();
            return queue;
        }

        private static void Preorder(WordCollection wordCollection, Node root)
        {
            if (root == null) return;
            if (root.Left != null)
            {
                root.Left.Code = root.Code + "0";
                if (root.Left.WordInfo != null)
                {
                    wordCollection.SetCode(root.Left.Word, root.Left.Code.ToCharArray());
                    SetPoint(wordCollection, root.Left.Word, root.Left.Code.Length, root, 1);
                }

            }

            if (root.Right != null)
            {
                root.Right.Code = root.Code + "1";
                if (root.Right.WordInfo != null)
                {
                    wordCollection.SetCode(root.Right.Word, root.Right.Code.ToCharArray());
                    SetPoint(wordCollection, root.Right.Word, root.Right.Code.Length, root, 1);

                }
            }
            Preorder(wordCollection, root.Right);
            Preorder(wordCollection, root.Left);
        }

        private static void SetPoint(WordCollection wordCollection, string word, int codeLength, Node root, int i)
        {
            wordCollection.SetPoint(word, codeLength - i, root.IndexOfLeafNodeThisInteriorNodePretendsToBe.Value);

            if (codeLength - i == 0 || root.Parent == null)
            {
                return;
            }

            SetPoint(wordCollection, word, codeLength, root.Parent, ++i);
        }

        private class Node : IComparable<Node>
        {
            public Node Parent { get; set; }
            public string Word { get; set; }
            public string Code { get; set; }
            public WordInfo WordInfo { get; set; }
            public Node Left { get; set; }
            public Node Right { get; set; }
            public long Frequency { get; set; }
            public long? IndexOfLeafNodeThisInteriorNodePretendsToBe { get; set; }

            public int CompareTo(Node other)
            {
                return Comparer<long>.Default.Compare(Frequency, other.Frequency);

            }
        }
    }
}