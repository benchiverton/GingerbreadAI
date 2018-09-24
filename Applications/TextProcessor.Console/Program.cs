using CoreTweet.Streaming;
using Emotion.Detector;
using Emotion.Detector.Repositories;
using Emotion.Detector.Repositories.Cache;
using log4net;
using StructureMap;
using System;
using System.IO;
using System.Reflection;
using System.Xml;
using TwitterProcessor.Console.Data;
using TwitterProcessor.Console.Observers;
using TwitterProcessor.Console.TwitterAuthorisers;

namespace TwitterProcessor.Console
{
    class Program
    {
        private static readonly ILog _logger = LogManager.GetLogger(typeof(Program));

        public static void Main(string[] args)
        {
            ConfigureLog4Net();

            var container = new Container(registry =>
            {
                registry.For<ILog>().Use(_logger).Singleton();
                registry.For<WordCache>().Use<WordCache>().Singleton();
                registry.For<WordRepository>().Use<WordRepository>();
                registry.For<NegationManager>().Use<NegationManager>().Ctor<string>().Is("Resources/Negations.txt");
                registry.For<EmotionDetector>().Use<EmotionDetector>();
                registry.For<ITwitterAuthoriser>().Use<TwitterAuthoriserConsole>();
                registry.For<ITweetObserver<StreamingMessage, Tweet>>().Use<TweetObserver>();
                registry.For<TweetListener>().Use<TweetListener>();
                registry.For<ProcessEngine>().Use<ProcessEngine>();
            });

            var processEngine = container.GetInstance<ProcessEngine>();

            processEngine.Initialise("brexit");
            processEngine.Start();

            while (true)
            {

            }
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
