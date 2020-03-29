using System.IO;
using System.IO.Compression;
using System.Net;

namespace NeuralNetwork.Test.Helpers
{
    public static class DownloadHelpers
    {
        public static FileInfo DownloadAndDecompressGzFile(string fileUrl, string saveLocation)
        {
            DownloadFile(fileUrl, $"{saveLocation}.gz");

            var fileInfo = new FileInfo($"{saveLocation}.gz");
            string newFileName;
            using (var originalFileStream = fileInfo.OpenRead())
            {
                var currentFileName = fileInfo.FullName;
                newFileName = currentFileName.Remove(currentFileName.Length - fileInfo.Extension.Length);
                using var decompressedFileStream = File.Create(newFileName);
                using var decompressionStream = new GZipStream(originalFileStream, CompressionMode.Decompress);
                decompressionStream.CopyTo(decompressedFileStream);
            }

            File.Delete($"{saveLocation}.gz");

            return new FileInfo(newFileName);
        }

        public static void DownloadFile(string fileUrl, string saveLocation)
        {
            using var webClient = new WebClient { Proxy = new WebProxy() };
            webClient.DownloadFile(fileUrl, saveLocation);
        }
    }
}
