using System.Diagnostics;
using Xunit;

namespace GingerbreadAI.NeuralNetwork.Test
{
    public sealed class RunnableInDebugOnlyAttribute : FactAttribute
    {
        public RunnableInDebugOnlyAttribute()
        {
            if (!Debugger.IsAttached)
            {
                Skip = "Only running in interactive mode.";
            }
        }
    }
}