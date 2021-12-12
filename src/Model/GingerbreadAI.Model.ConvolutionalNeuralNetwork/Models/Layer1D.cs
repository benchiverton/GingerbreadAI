using GingerbreadAI.Model.NeuralNetwork.ActivationFunctions;
using GingerbreadAI.Model.NeuralNetwork.InitialisationFunctions;
using GingerbreadAI.Model.NeuralNetwork.Models;

namespace GingerbreadAI.Model.ConvolutionalNeuralNetwork.Models;

public class Layer1D : Layer
{
    public int Size { get; }

    public Layer1D(int size, Layer[] previousGroups,
        ActivationFunctionType activationFunctionType, InitialisationFunctionType initialisationFunctionType)
    : base(size, previousGroups, activationFunctionType, initialisationFunctionType) =>
        Size = size;
}
