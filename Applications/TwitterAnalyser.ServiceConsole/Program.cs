using log4net;
using NServiceBus;
using NServiceBus.Features;
using System;
using System.IO;
using System.Reflection;
using System.Xml;
using TweetListener.Events;
using TwitterAnalyser.ServiceConsole.Handlers;

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

            // register stuff nd run
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

        private static void ConfigureNServiceBus(string topic)
        {
            var endpointConfiguration = new EndpointConfiguration(Assembly.GetExecutingAssembly().GetName().Name);
            endpointConfiguration.SendFailedMessagesTo("error");
            endpointConfiguration.UseSerialization<NewtonsoftSerializer>();
            endpointConfiguration.DisableFeature<AutoSubscribe>();
            endpointConfiguration.EnableInstallers();

            endpointConfiguration.RegisterComponents(r =>
                r.ConfigureComponent<TweetReceivedHandler>(DependencyLifecycle.InstancePerCall)
            );

            var transport = endpointConfiguration.UseTransport<AzureServiceBusTransport>();
            transport.ConnectionString(Environment.GetEnvironmentVariable("serviceBusConnectionString"));
            transport.TopicName($"SentimentAnalyser.Twitter.{topic.Replace(" ", string.Empty)}");

            var endPoint = Endpoint.Start(endpointConfiguration).GetAwaiter().GetResult();
            endPoint.Subscribe<TweetReceived>().GetAwaiter().GetResult();
        }
    }
}
