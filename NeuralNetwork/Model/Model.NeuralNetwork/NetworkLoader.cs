using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using Model.NeuralNetwork.Models;

namespace Model.NeuralNetwork
{
    public static class NetworkLoader
    {
        public static Layer LoadNetwork(string location)
        {
            using (var byteStream = File.OpenRead(location))
            {
                using (var memoryStream = new MemoryStream())
                {
                    byteStream.CopyTo(memoryStream);
                    memoryStream.Position = 0;

                    var formatter = new BinaryFormatter();
                    return (Layer)formatter.Deserialize(memoryStream);
                }
            }
        }
    }
}
