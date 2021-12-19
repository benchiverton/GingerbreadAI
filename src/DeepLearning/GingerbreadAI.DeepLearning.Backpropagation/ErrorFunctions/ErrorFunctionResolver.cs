using System;

namespace GingerbreadAI.DeepLearning.Backpropagation.ErrorFunctions;

public static class ErrorFunctionResolver
{
    /// <summary>
    /// Returns the differential of the error function supplied
    /// Function returned is of the following signature: (target, actual) => differential of error
    /// </summary>
    /// <param name="errorFunctionType"></param>
    /// <returns></returns>
    public static Func<double, double, double> ResolveErrorFunctionDifferential(ErrorFunctionType errorFunctionType) => errorFunctionType switch
    {
        ErrorFunctionType.MSE => (target, actual) => actual - target,
        ErrorFunctionType.CrossEntropy => (target, actual) => (actual - target) * Math.Min(1 / ((1 - actual) * actual), 1000000),
        _ => throw new ArgumentOutOfRangeException(
            nameof(errorFunctionType),
            errorFunctionType,
            "This error function type is not yet supported. Please use a different error function type.")
    };
}
