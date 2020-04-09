using System;
using System.Collections.Generic;
using System.Linq;

namespace GingerbreadAI.NLP.Word2Vec
{
    public class HuffmanTree
    {
        /**
         * ======== CreateBinaryTree ========
         * Create binary Huffman tree using the word counts.
         * Frequent words will have short unique binary codes.
         * Huffman encoding is used for lossless compression.
         * The vocab_word structure contains a field for the 'code' for the word.
         */
        private WordCollection _wordCollection;

        public void Create(WordCollection wordCollection)
        {
            _wordCollection = wordCollection;
            var queue = GetQueueSortedByWordFrequencyAscending(wordCollection);
            IterateQueue(wordCollection, queue);
            var root = queue.Single();
            root.Code = "";
            Preorder(root);
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
                    {Frequency = word.Value.Count, WordInfo = word.Value, Word = word.Key})
                .OrderBy(y => y.Frequency).ToList();
            return queue;
        }

        private void Preorder(Node root)
        {
            if (root == null) return;
            if (root.Left != null)
            {
                root.Left.Code = root.Code + "0";
                if (root.Left.WordInfo != null)
                {
                    _wordCollection.SetCode(root.Left.Word, root.Left.Code.ToCharArray());
                    SetPoint(root.Left.Word, root.Left.Code.Length, root, 1);
                }

            }

            if (root.Right != null)
            {
                root.Right.Code = root.Code + "1";
                if (root.Right.WordInfo != null)
                {
                    _wordCollection.SetCode(root.Right.Word, root.Right.Code.ToCharArray());
                    SetPoint(root.Right.Word, root.Right.Code.Length, root, 1);

                }
            }
            Preorder(root.Right);
            Preorder(root.Left);
        }

        private void SetPoint(string word, int codeLength, Node root, int i)
        {
            _wordCollection.SetPoint(word, codeLength - i, root.IndexOfLeafNodeThisInteriorNodePretendsToBe.Value);

            if (codeLength - i == 0 || root.Parent == null)
            {
                return;
            }

            SetPoint(word, codeLength, root.Parent, ++i);
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