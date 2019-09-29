namespace NeuralNetwork
{
    using NeuralNetwork.Models;
    using System.IO;
    using System.Runtime.Serialization.Formatters.Binary;

    public static class NetworkLoader
    {
        public static Layer LoadNetwork(string location)
        {
            using (var byteStream = File.OpenRead(location))
            {
                var memoryStream = new MemoryStream();
                byteStream.CopyTo(memoryStream);
                memoryStream.Position = 0;

                var formatter = new BinaryFormatter();
                return (Layer)formatter.Deserialize(memoryStream);
            }
        }
    }
}
