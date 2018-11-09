using System;
using System.IO;
using System.Reflection;
using System.Xml;
using Emotion.Detector.Lexicons;
using Emotion.Detector.Lexicons.Detectors;
using Emotion.Detector.Lexicons.Interfaces;
using Emotion.Detector.Lexicons.Repositories;
using Emotion.Detector.Lexicons.Repositories.Cache;
using log4net;
using NServiceBus;
using NServiceBus.Features;
using StructureMap;
using TweetAnalyserV1.ServiceConsole.Caches;
using TweetAnalyserV1.ServiceConsole.Handlers;
using TweetAnalyserV1.ServiceConsole.Persisters;
using TweetListener.Events;

namespace TweetAnalyserV1.ServiceConsole
{
    public class Program
    {
        private static readonly ILog Logger = LogManager.GetLogger(typeof(Program));

        public static void Main(string[] args)
        {
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

            ConfigureAndStartEndpoint(container);

            Logger.Info("Tweet Analyser v1 started!");
            while (true)
            {

            }
        }

        private static void ConfigureLog4Net()
        {
            GlobalContext.Properties["LogName"] = typeof(Program).Assembly.GetName().Name;

            var log4NetConfig = new XmlDocument();
#if DEBUG
            log4NetConfig.Load(File.OpenRead("log4net_debug.config"));
#else
            log4NetConfig.Load(File.OpenRead("log4net.config"));
#endif
            var repo = LogManager.CreateRepository(Assembly.GetEntryAssembly(), typeof(log4net.Repository.Hierarchy.Hierarchy));
            log4net.Config.XmlConfigurator.Configure(repo, log4NetConfig["log4net"]);
        }

        private static void ConfigureAndStartEndpoint(Container container)
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
            transport.TopicName("Twitter.SentimentAnalyser");
            transport.Transactions(TransportTransactionMode.ReceiveOnly);

            var endPoint = Endpoint.Start(endpointConfiguration).GetAwaiter().GetResult();
            endPoint.Subscribe<TweetReceived>().GetAwaiter().GetResult();
        }
    }
}
