using Emotion.Detector;
using Emotion.Detector.Detectors;
using Emotion.Detector.Interfaces;
using Emotion.Detector.Repositories;
using Emotion.Detector.Repositories.Cache;
using log4net;
using NServiceBus;
using NServiceBus.Features;
using StructureMap;
using System;
using System.IO;
using System.Reflection;
using System.Xml;
using TweetListener.Events;
using TwitterAnalyser.ServiceConsole.Caches;
using TwitterAnalyser.ServiceConsole.Handlers;
using TwitterAnalyser.ServiceConsole.Persisters;

namespace TwitterAnalyser.ServiceConsole
{
    class Program
    {
        private static readonly ILog Logger = LogManager.GetLogger(typeof(Program));

        static void Main(string[] args)
        {
            if (args.Length != 1)
                throw new ArgumentException("Please supply an argument specifying which topic you wish to stream tweets from.");

            var topic = args[0];

            ConfigureLog4Net();

            var container = new Container(registry =>
            {
                registry.For<ILog>().Use(Logger);
                registry.For<WordCache>().Use<WordCache>().Singleton();
                registry.For<WordRepository>().Use<WordRepository>().Singleton();
                registry.For<NegationManager>().Use<NegationManager>().Ctor<string>().Is("Resources/Negations.txt").Singleton();
                registry.For<IEmotionDetector>().Use<EmotionDetector>();
                registry.For<EmotionPersister>().Use<EmotionPersister>();
                registry.For<TweetCache>().Use<TweetCache>().Singleton();
                registry.For<TweetReceivedHandler>().Use<TweetReceivedHandler>(); // contains cache
            });

            ConfigureAndStartEndpoint(container, args[0]);

            Logger.Info("Tweet Analyser v1 started!");
            while (true)
            {

            }
        }

        private static void ConfigureLog4Net()
        {
            GlobalContext.Properties["LogName"] = typeof(Program).Assembly.GetName().Name;

            var log4NetConfig = new XmlDocument();
            log4NetConfig.Load(File.OpenRead("log4net.config"));
            var repo = LogManager.CreateRepository(Assembly.GetEntryAssembly(),
                       typeof(log4net.Repository.Hierarchy.Hierarchy));
            log4net.Config.XmlConfigurator.Configure(repo, log4NetConfig["log4net"]);
        }

        private static void ConfigureAndStartEndpoint(Container container, string topic)
        {
            var endpointConfiguration = new EndpointConfiguration(Assembly.GetExecutingAssembly().GetName().Name);
            endpointConfiguration.SendFailedMessagesTo("error");
            endpointConfiguration.UseSerialization<NewtonsoftSerializer>();
            endpointConfiguration.DisableFeature<AutoSubscribe>();
            endpointConfiguration.EnableInstallers();

            endpointConfiguration.UseContainer<StructureMapBuilder>(
            customizations: customizations =>
            {
                customizations.ExistingContainer(container);
            });

            var transport = endpointConfiguration.UseTransport<AzureServiceBusTransport>();
            transport.ConnectionString(Environment.GetEnvironmentVariable("serviceBusConnectionString"));
            transport.TopicName($"SentimentAnalyser.Twitter.{topic.Replace(" ", string.Empty)}");

            var endPoint = Endpoint.Start(endpointConfiguration).GetAwaiter().GetResult();
            endPoint.Subscribe<TweetReceived>().GetAwaiter().GetResult();
        }
    }
}
