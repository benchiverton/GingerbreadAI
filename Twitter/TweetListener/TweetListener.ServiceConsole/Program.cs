using CoreTweet;
using log4net;
using NServiceBus;
using NServiceBus.Features;
using StructureMap;
using System;
using System.IO;
using System.Reflection;
using System.Text.RegularExpressions;
using System.Xml;
using TweetListener.Engine;
using TweetListener.Engine.Observers;
using TweetListener.Engine.Persisters;
using TweetListener.ServiceConsole.TwitterAuthorisers;

namespace TweetListener.ServiceConsole
{
    public class Program
    {
        private static readonly ILog Logger = LogManager.GetLogger(typeof(Program));
        private static string _topic;

        public static void Main(string[] args)
        {
            if (args.Length != 1)
                throw new ArgumentException("Please supply an argument specifying which topic you wish to stream tweets from.");

            _topic = args[0];

            ConfigureLog4Net();

            var container = new Container(registry =>
            {
                registry.For<ILog>().Use(Logger).Singleton();
                registry.For<Tokens>().Use(GetTwitterTokens()).Singleton();
                registry.For<ITweetObserver>().Use<TweetObserver>().Ctor<int>().Is(1000);
                registry.For<ITweetPersister>().Use<TweetPersister>();
                registry.For<IEndpointInstance>().Use(ConfigureNServiceBus());
                registry.For<HistoricTweetCache>().Use<HistoricTweetCache>().Singleton();
                registry.For<TweetProcessor>().Use<TweetProcessor>();
                registry.For<TweetStreamer>().Use<TweetStreamer>();
                registry.For<ProcessEngine>().Use<ProcessEngine>();
            });

            var processEngine = container.GetInstance<ProcessEngine>();

            processEngine.Initialise(_topic);
            processEngine.Start();

            while (true)
            {

            }
        }

        private static void ConfigureLog4Net()
        {
            GlobalContext.Properties["LogName"] = $"{typeof(Program).Assembly.GetName().Name}.{new Regex("[^a-zA-Z0-9]").Replace(_topic, "")}";

            var log4NetConfig = new XmlDocument();
#if DEBUG
            log4NetConfig.Load(File.OpenRead("log4net_debug.config"));
#else
            log4NetConfig.Load(File.OpenRead("log4net.config"));
#endif
            var repo = LogManager.CreateRepository(Assembly.GetEntryAssembly(), typeof(log4net.Repository.Hierarchy.Hierarchy));
            log4net.Config.XmlConfigurator.Configure(repo, log4NetConfig["log4net"]);
        }

        private static IEndpointInstance ConfigureNServiceBus()
        {
            var endpointConfiguration = new EndpointConfiguration(Assembly.GetExecutingAssembly().GetName().Name);
            endpointConfiguration.SendFailedMessagesTo("error");
            endpointConfiguration.UseSerialization<NewtonsoftSerializer>();
            endpointConfiguration.DisableFeature<AutoSubscribe>();

            var transport = endpointConfiguration.UseTransport<AzureServiceBusTransport>();
            transport.ConnectionString(Environment.GetEnvironmentVariable("serviceBusConnectionString"));
            transport.TopicName("Twitter.SentimentAnalyser");

            return Endpoint.Start(endpointConfiguration).GetAwaiter().GetResult();
        }

        private static Tokens GetTwitterTokens()
        {
            Tokens token = null;
            try
            {
                var consumerKey = Environment.GetEnvironmentVariable("twitterConsumerKey");
                var consumerSecret = Environment.GetEnvironmentVariable("twitterConsumerSecret");
                var twitterAuthoriser = new TwitterAuthoriserConsole();

                var session = OAuth.AuthorizeAsync(consumerKey, consumerSecret).GetAwaiter().GetResult();
                var pincode = twitterAuthoriser.GetPinCode(session.AuthorizeUri);
                token = session.GetTokensAsync(pincode).GetAwaiter().GetResult();
            }
            catch (Exception e)
            {
                Logger.Error($"Something went wrong whilst connecting to Twitter.");
                Logger.Debug($"Message:\r\n{e.Message}\r\nStack trace:\r\n{e.StackTrace}");
            }
            return token;
        }
    }
}
