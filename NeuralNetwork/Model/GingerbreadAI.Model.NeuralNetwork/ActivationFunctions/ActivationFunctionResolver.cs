using System;

namespace GingerbreadAI.Model.NeuralNetwork.ActivationFunctions
{
    public static class ActivationFunctionResolver
    {
        public static (Func<double, double> activationFunction, Func<double, double> activationFunctionDifferential) ResolveActivationFunctions(ActivationFunctionType activationFunctionType)
        {
            Func<double, double> activationFunction;
            Func<double, double> activationFunctionDifferential;
            switch (activationFunctionType)
            {
                case ActivationFunctionType.Linear:
                    activationFunction = input => input;
                    activationFunctionDifferential = _ => 1;
                    break;
                case ActivationFunctionType.RELU:
                    activationFunction = input => input > 0 ? input : 0;
                    activationFunctionDifferential = output => output > 0 ? 1 : 0;
                    break;
                case ActivationFunctionType.Sigmoid:
                    activationFunction = input => 1 / (1 + Math.Pow(Math.E, -input));
                    activationFunctionDifferential = output => output * (1 - output);
                    break;
                case ActivationFunctionType.Tanh:
                    activationFunction = input => (Math.Pow(Math.E, 2 * input) - 1) / (Math.Pow(Math.E, 2 * input) + 1);
                    activationFunctionDifferential = output => 1 - Math.Pow(output, 2);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(
                        nameof(activationFunctionType),
                        activationFunctionType,
                        "This activation function type is not yet supported. Please use a different activation function type.");
            }

            return (activationFunction, activationFunctionDifferential);
        }
    }
}
