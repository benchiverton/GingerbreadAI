namespace Backpropagation.Library
{
    public static class BackpropagationCalculations
    {
        public static double GetDeltaOutput(double actual, double target)
        {
            return -(target - actual) * actual * (1 - actual);
        }
    }
}
