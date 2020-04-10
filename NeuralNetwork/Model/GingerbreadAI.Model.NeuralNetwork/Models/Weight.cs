namespace GingerbreadAI.Model.NeuralNetwork.Models
{
    public class Weight
    {
        public Weight(double value)
        {
            Value = value;
        }

        public virtual double Value { get; private set; }

        public virtual void Adjust(double change)
        {
            Value += change;
        }
    }
}
