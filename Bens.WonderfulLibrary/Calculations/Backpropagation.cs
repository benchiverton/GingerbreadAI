namespace Bens.WonderfulLibrary.Calculations
{
    public static class Backpropagation
    {
        public static double GetDeltaOutput(double actual, double target)
        {
            return -(target - actual) * actual * (1 - actual);
        }
    }
}
