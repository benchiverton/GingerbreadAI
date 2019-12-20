using System;

namespace Library.Computations {
    public static class NetworkCalculations {

        public static double RandomInitialisation (Random rand) {
            return (2 * rand.NextDouble ()) - 1;
        }

        public static double GetWeightedInitialisation (Random rand, int feedingNodes) {
            return (2 * rand.NextDouble () / Math.Sqrt (feedingNodes)) - (1 / Math.Sqrt (feedingNodes));
        }
    }
}