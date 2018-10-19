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

        public static void Main(string[] args)
        {
            if(args.Length != 1)
                throw new ArgumentException("Please supply an argument specifying which topic you wish to stream tweets from.");

            var topic = args[0];

            ConfigureLog4Net();

            var container = new Container(registry =>
            {
                registry.For<ILog>().Use(Logger).Singleton();
                registry.For<Tokens>().Use(GetTwitterTokens()).Singleton();
                registry.For<ITweetObserver>().Use<TweetObserver>().Ctor<int>().Is(1000);
                registry.For<ITweetPersister>().Use<TweetPersister>();
                registry.For<IEndpointInstance>().Use(ConfigureNServiceBus(topic));
                registry.For<HistoricTweetCache>().Use<HistoricTweetCache>().Singleton();
                registry.For<TweetProcessor>().Use<TweetProcessor>();
                registry.For<TweetStreamer>().Use<TweetStreamer>();
                registry.For<ProcessEngine>().Use<ProcessEngine>();
            });

            var processEngine = container.GetInstance<ProcessEngine>();

            processEngine.Initialise(topic);
            processEngine.Start();

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

        private static IEndpointInstance ConfigureNServiceBus(string topic)
        {
            var endpointConfiguration = new EndpointConfiguration(Assembly.GetExecutingAssembly().GetName().Name);
            endpointConfiguration.SendFailedMessagesTo("error");
            endpointConfiguration.UseSerialization<NewtonsoftSerializer>();
            endpointConfiguration.DisableFeature<AutoSubscribe>();
            endpointConfiguration.EnableInstallers();

            var transport = endpointConfiguration.UseTransport<AzureServiceBusTransport>();
            transport.ConnectionString(Environment.GetEnvironmentVariable("serviceBusConnectionString"));
            transport.TopicName($"SentimentAnalyser.Twitter.{new Regex("[^a-zA-Z0-9]").Replace(topic, "")}");

            return Endpoint.Start(endpointConfiguration).GetAwaiter().GetResult();
        }

        private static Tokens GetTwitterTokens()
        {
            Tokens token = null;
            OAuth.OAuthSession session;
            try
            {
                var consumerKey = Environment.GetEnvironmentVariable("twitterConsumerKey", EnvironmentVariableTarget.User);
                var consumerSecret = Environment.GetEnvironmentVariable("twitterConsumerSecret", EnvironmentVariableTarget.User);
                var twitterAuthoriser = new TwitterAuthoriserConsole();

                session = OAuth.AuthorizeAsync(consumerKey, consumerSecret).GetAwaiter().GetResult();
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
