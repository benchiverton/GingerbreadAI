using System;
using System.Collections.Generic;
using System.Text;

namespace DeepLearning
{
    public static class BackpropogationCalculations
    {
        public static double GetDeltaOutput(double actual, double target)
        {
            return -(target - actual) * actual * (1 - actual) * actual;
        }
    }
}
