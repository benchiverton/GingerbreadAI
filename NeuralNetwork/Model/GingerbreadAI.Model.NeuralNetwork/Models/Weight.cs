namespace Model.NeuralNetwork.Models
{
    public class Weight
    {
        public Weight(double value)
        {
            Value = value;
        }

        public virtual double Value { get; set; }
    }
}
