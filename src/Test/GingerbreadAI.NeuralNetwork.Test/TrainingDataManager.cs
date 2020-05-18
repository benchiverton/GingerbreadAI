using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using MNIST.IO;

namespace GingerbreadAI.NeuralNetwork.Test
{
    public static class TrainingDataManager
    {
        private const string TrainingDataDir = "TrainingData";

        private const string MNISTHandwrittenNumbersDir = "MNISTHandwrittenNumbersData";
        private const string BlogAuthorshipCorpusDir = "BlogAuthorshipCorpus";

        // http://yann.lecun.com/exdb/mnist/
        public static IEnumerable<(double[] image, int label)> GetMNISTHandwrittenNumbers(string labelFileName, string imageFileName)
        {
            var dataDirectory = $"{TrainingDataDir}/{MNISTHandwrittenNumbersDir}";

            EnsureMNISTHandwrittenNumbersDataExists(dataDirectory);

            var trainingDataSet = FileReaderMNIST.LoadImagesAndLables($"{dataDirectory}/{labelFileName}", $"{dataDirectory}/{imageFileName}");

            foreach (var trainingData in trainingDataSet)
            {
                var trainingDataAsDoubleArray = new double[784];
                var trainingDataAsDouble = trainingData.AsDouble();
                for (var i = 0; i < 28; i++)
                {
                    for (var j = 0; j < 28; j++)
                    {
                        trainingDataAsDoubleArray[j + 28 * i] = trainingDataAsDouble[i, j];
                    }
                }
                yield return (trainingDataAsDoubleArray, trainingData.Label);
            }
        }

        // http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm
        public static IEnumerable<FileInfo> GetBlogAuthorshipCorpusFiles()
        {
            var dataDirectory = $"{TrainingDataDir}/{BlogAuthorshipCorpusDir}";

            EnsureBlogAuthorshipCorpusDataExists(dataDirectory);

            return new DirectoryInfo($"{dataDirectory}/blogs").EnumerateFiles();
        }
        
        /// <summary>
        /// Returns a file that contains the string '1 2 3 4 5 6 7 8 9 10' on 10,000 lines.
        /// </summary>
        public static FileInfo GetAlphabetFile()
        {
            return new FileInfo($"{TrainingDataDir}/Alphabet/Alphabet.txt");
        }

        private static void EnsureMNISTHandwrittenNumbersDataExists(string dataDirectory)
        {
            if (!Directory.Exists(dataDirectory))
            {
                Directory.CreateDirectory(dataDirectory);
            }

            var directoryFiles = new DirectoryInfo(dataDirectory).EnumerateFiles().Select(f => f.Name).ToArray();

            if (!directoryFiles.Contains("train-images-idx3-ubyte.gz"))
            {
                DownloadFile("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", $"{dataDirectory}/train-images-idx3-ubyte.gz");
            }
            if (!directoryFiles.Contains("train-labels-idx1-ubyte.gz"))
            {
                DownloadFile("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", $"{dataDirectory}/train-labels-idx1-ubyte.gz");
            }
            if (!directoryFiles.Contains("t10k-images-idx3-ubyte.gz"))
            {
                DownloadFile("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", $"{dataDirectory}/t10k-images-idx3-ubyte.gz");
            }
            if (!directoryFiles.Contains("t10k-images-idx1-ubyte.gz"))
            {
                DownloadFile("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", $"{dataDirectory}/t10k-labels-idx1-ubyte.gz");
            }
        }

        private static void EnsureBlogAuthorshipCorpusDataExists(string dataDirectory)
        {
            if (!Directory.Exists(dataDirectory))
            {
                Directory.CreateDirectory(dataDirectory);
            }

            var trainingDataDirectory = new DirectoryInfo(dataDirectory);

            if (!trainingDataDirectory.EnumerateFiles().Select(f => f.Name).Contains("blogs.zip"))
            {
                DownloadFile("http://www.cs.biu.ac.il/~koppel/blogs/blogs.zip", $"{dataDirectory}/blogs.zip");
            }

            if (!trainingDataDirectory.EnumerateDirectories().Select(d => d.Name).Contains("blogs"))
            {
                ZipFile.ExtractToDirectory($"{dataDirectory}/blogs.zip", dataDirectory);
            }
        }

        private static void DownloadFile(string fileUrl, string saveLocation)
        {
            using var webClient = new WebClient { Proxy = new WebProxy() };
            webClient.DownloadFile(fileUrl, saveLocation);
        }
    }
}
