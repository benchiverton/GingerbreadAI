using System;

namespace GingerbreadAI.DeepLearning.Backpropagation.ErrorFunctions
{
    public static class ErrorFunctionResolver
    {
        /// <summary>
        /// Returns the differential of the error function supplied
        /// Function returned is of the following signature: (target, actual) => differential of error
        /// </summary>
        /// <param name="errorFunctionType"></param>
        /// <returns></returns>
        public static Func<double, double, double> ResolveErrorFunctionDifferential(ErrorFunctionType errorFunctionType)
        {
            return errorFunctionType switch
            {
                ErrorFunctionType.MSE => (target, actual) => actual - target,
                ErrorFunctionType.CrossEntropy => (target, actual) => actual % 1 < 0.0000001 ? 0 : -(target / actual) + (1 - target) / (1 - actual),
                _ => throw new ArgumentOutOfRangeException(
                    nameof(errorFunctionType),
                    errorFunctionType,
                    "This error function type is not yet supported. Please use a different error function type.")
            };
        }
    }
}
