using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Nodes
{
    public static class NodeLayerExtension
    {
        /// <summary>
        /// Extension method to do a foreach loop with an index
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="ie"></param>
        /// <param name="action"></param>
        public static void Each<T>(this IEnumerable<T> ie, Action<T, int> action)
        {
            var i = 0;
            foreach (var e in ie) action(e, i++);
        }
    }
}
