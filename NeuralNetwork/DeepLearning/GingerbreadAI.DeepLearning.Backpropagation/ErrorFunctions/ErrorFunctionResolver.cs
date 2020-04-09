using System;
using GingerbreadAI.Model.NeuralNetwork.ActivationFunctions;

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
            Func<double, double, double> errorFunctionDifferential;
            switch (errorFunctionType)
            {
                case ErrorFunctionType.MSE:
                    errorFunctionDifferential = (target, actual) => actual - target;
                    break;
                case ErrorFunctionType.CrossEntropy:
                    errorFunctionDifferential = (target, actual) => Math.Abs(actual) < 0.00001 ? 0 : - (target / actual) + (1 - target) / (1 - actual);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(
                        nameof(errorFunctionType),
                        errorFunctionType,
                        "This error function type is not yet supported. Please use a different error function type.");
            }

            return errorFunctionDifferential;
        }
    }
}
