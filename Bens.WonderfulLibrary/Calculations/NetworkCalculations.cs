namespace Bens.WonderfulLibrary.Calculations
{
    using System;

    public static class NetworkCalculations
    {
        public static double LogisticFunction(double value)
        {
            return 1 / (1 + Math.Pow(Math.E, -value));
        }

        public static double LogisticFunctionDifferential(double value) {
        {
            return value * (1 - value);
        } }
    }
}