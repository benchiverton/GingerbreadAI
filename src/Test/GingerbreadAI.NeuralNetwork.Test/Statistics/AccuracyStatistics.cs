using System;
using System.Text;

namespace GingerbreadAI.NeuralNetwork.Test.Statistics;

public static class AccuracyStatistics
{
    // https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    // smaller value returned => values closer to target
    public static double CalculateKolmogorovStatistic(double[] values, double[] target)
    {
        ValidateData(values, target);

        var result = 0d;

        for (var i = 0; i < values.Length; i++)
        {
            var difference = Math.Abs(values[i] - target[i]);
            if (difference > result)
            {
                result = difference;
            }
        }

        return result;
    }

    private static void ValidateData(double[] values, double[] target)
    {
        var errorMessage = new StringBuilder();
        var shouldThrowException = false;

        if (values == null)
        {
            errorMessage.AppendLine("'values' cannot be null");
            shouldThrowException = true;
        }
        if (target == null)
        {
            errorMessage.AppendLine("'target' cannot be null");
            shouldThrowException = true;
        }

        if (values?.Length != target?.Length)
        {
            errorMessage.AppendLine("List 'values' must be the same length as the 'targets'");
            shouldThrowException = true;
        }

        if (shouldThrowException)
        {
            throw new Exception(errorMessage.ToString());
        }
    }
}
