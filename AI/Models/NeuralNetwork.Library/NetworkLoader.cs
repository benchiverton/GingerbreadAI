using NeuralNetwork.Data;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace NeuralNetwork.Library
{
    public static class NetworkLoader
    {
        public static Layer LoadNetwork(string location)
        {
            var byteStream = File.OpenRead(location);

            var memoryStream = new MemoryStream();
            byteStream.CopyTo(memoryStream);
            memoryStream.Position = 0;

            var formatter = new BinaryFormatter();
            return (Layer)formatter.Deserialize(memoryStream);
        }
    }
}
