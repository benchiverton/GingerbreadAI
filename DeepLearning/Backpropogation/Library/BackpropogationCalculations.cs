using System;
using System.Collections.Generic;
using System.Text;

namespace Backpropogation.Library
{
    public static class BackpropogationCalculations
    {
        public static double GetDeltaOutput(double actual, double target)
        {
            return -(target - actual) * actual * (1 - actual) * actual;
        }
    }
}
