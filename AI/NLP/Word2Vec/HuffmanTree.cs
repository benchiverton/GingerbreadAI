using System;
using System.Collections.Generic;
using System.Linq;

namespace Word2Vec
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
            var sortedByLowestCount = wordCollection.ToArray();
            var queue = sortedByLowestCount.Select(word => new Node
            { Frequency = word.Value.Count, WordInfo = word.Value, Word = word.Key })
                .OrderBy(y => y.Frequency).ToList();

            var count = new long[wordCollection.GetNumberOfUniqueWords() * 2 + 1];
            var keys = wordCollection.GetWords().ToArray();

            for (var a = 0; a < wordCollection.GetNumberOfUniqueWords(); a++)
                count[a] = wordCollection.GetOccurrenceOfWord(keys[a]);
            for (var a = wordCollection.GetNumberOfUniqueWords(); a < wordCollection.GetNumberOfUniqueWords() * 2; a++)
                count[a] = (long)1e15;
            var numberOfNoneLeafNodes = 0;
            for (var a = 0; a < wordCollection.GetNumberOfUniqueWords() - 1; a++)
            {
                var node = new Node
                {
                    Left = queue.First(), Right = queue.Skip(1).First(),
                    IndexOfLeafNodeThisNoneLeafNodePretendsToBe = numberOfNoneLeafNodes
                };
                numberOfNoneLeafNodes++;
                node.Left.Parent = node;
                node.Right.Parent = node;
                node.Frequency = node.Left.Frequency + node.Right.Frequency;
                queue.Remove(node.Left);
                queue.Remove(node.Right);
                var index = queue.BinarySearch(node, new FrequencyComparer());
                if (index >= 0)
                {
                    queue.Insert(index, node);
                }
                else
                {
                    queue.Insert(~index, node);
                }
            }

            var root = queue.Single();
            root.Code = "";
            Preorder(root);
            GC.Collect();
        }


        private class FrequencyComparer : IComparer<Node>
        {
            public int Compare(Node x, Node y)
            {
                return Comparer<long>.Default.Compare(x.Frequency, y.Frequency);
            }
        }

        private void Preorder(Node root)
        {
            if (root != null)
            {
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
        }

        private void SetPoint(string word, int codeLength, Node root, int i)
        {
            _wordCollection.SetPoint(word, codeLength - i, root.IndexOfLeafNodeThisNoneLeafNodePretendsToBe.Value);

            if (codeLength - i == 0 || root.Parent == null)
            {
                return;
            }

            SetPoint(word, codeLength, root.Parent, ++i);
        }
        
        private class Node
        {
            public Node Parent { get; set; }
            public string Word { get; set; }
            public string Code { get; set; }
            public WordInfo WordInfo { get; set; }
            public Node Left { get; set; }
            public Node Right { get; set; }
            public long Frequency { get; set; }
            public long? IndexOfLeafNodeThisNoneLeafNodePretendsToBe { get; set; }
        }
    }
}