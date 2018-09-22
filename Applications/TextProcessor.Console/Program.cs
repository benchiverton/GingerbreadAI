using Emotion.Detector;
using Emotion.Detector.Repository;
using Emotion.Detector.Repository.Cache;
using log4net;
using StructureMap;
using System;
using System.IO;
using System.Reflection;
using System.Xml;

namespace TextProcessor.Console
{
    class Program
    {
        private static readonly ILog _logger = LogManager.GetLogger(typeof(Program));

        public static void Main(string[] args)
        {
            // environment variable as this is planned to be a docker container
            Environment.SetEnvironmentVariable("wordRepositoryConnectionString",
                "data source=localhost;database=TextAnalysis;Integrated Security=SSPI;persist security info=True;");

            ConfigureLog4Net();

            var container = new Container(registry =>
            {
                registry.For<ILog>().Use(_logger).Singleton();
                registry.For<WordCache>().Use<WordCache>().Singleton();
                registry.For<WordRepository>().Use<WordRepository>();
                registry.For<EmotionDetector>().Use<EmotionDetector>();
                // need something to listen to NSB as TweetSharp doesn't run on .Net Core.
            });

            var emotionDetector = container.GetInstance<EmotionDetector>();

            var emotion4 = emotionDetector.Detect("There seems to be brilliant weather today although I'm not sure that it will remain this way");
        }

        private static void ConfigureLog4Net()
        {
            log4net.GlobalContext.Properties["LogName"] = typeof(Program).Assembly.GetName().Name;

            var log4netConfig = new XmlDocument();
            log4netConfig.Load(File.OpenRead("log4net.config"));
            var repo = log4net.LogManager.CreateRepository(Assembly.GetEntryAssembly(),
                       typeof(log4net.Repository.Hierarchy.Hierarchy));
            log4net.Config.XmlConfigurator.Configure(repo, log4netConfig["log4net"]);
        }
    }
}
