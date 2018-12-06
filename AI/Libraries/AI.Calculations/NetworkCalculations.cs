namespace AI.Calculations
{
    using System;

    public static class NetworkCalculations
    {
        public static double LogisticFunction(double value)
        {
            return 1 / (1 + Math.Pow(Math.E, -value));
        }

        public static double LogisticFunctionDifferential(double value)
        {
            return value * (1 - value);
        }

        public static double RandomInitialisation(Random rand)
        {
            return (2 * rand.NextDouble()) - 1;
        }

        public static double GetWeightedInitialisation(Random rand, int feedingNodes)
        {
            return (2 * rand.NextDouble() / Math.Sqrt(feedingNodes)) - (1 / Math.Sqrt(feedingNodes));
        }
    }
}