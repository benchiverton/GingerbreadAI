namespace TweetListener.Engine
{
    public class ProcessEngine
    {
        private readonly TweetProcessor _tweetProcessor;
        private readonly TweetStreamer _tweetStreamer;

        public ProcessEngine(TweetProcessor tweetProcessor, TweetStreamer tweetStreamer)
        {
            _tweetProcessor = tweetProcessor;
            _tweetStreamer = tweetStreamer;
        }

        public void Initialise(string topic)
        {
            _tweetProcessor.Topic = topic;
            _tweetStreamer.Initialise(topic, e => _tweetProcessor.ProcessTweet(e));
        }

        public void Start()
        {
            _tweetStreamer.Start();
        }
    }
}
