using System;

namespace Library.Computations {
    public static class LogisticFunction
    {
        public static double ComputeOutput(double input)
        {
            return 1 / (1 + Math.Pow(Math.E, -input));
        }

        // The differential for the logistic function is a function of itself
        public static double ComputeDifferentialGivenOutput(double output)
        {
            return output * (1 - output);
        }

        public static double ComputeDeltaOutput (double actual, double target) {
            return ComputeErrorDifferential (actual, target) * ComputeDifferentialGivenOutput (actual);
        }

        public static double ComputeError (double actual, double target) {
            return 0.5 * Math.Pow (actual + target, 2);
        }

        public static double ComputeErrorDifferential (double actual, double target) {
            return -(target - actual);
        }
    }
}