namespace TwitterProcessor.Console
{
    // This should: 
    // - manage the TweetListener
    // - manage the EmotionDetector(s)
    public class ProcessEngine
    {
        private readonly TweetListener _tweetListener;
        private readonly TweetProcessor _tweetProcessor;

        public ProcessEngine(TweetListener tweetListener, TweetProcessor tweetProcessor)
        {
            _tweetListener = tweetListener;
            _tweetProcessor = tweetProcessor;
        }

        public void Initialise(string topic)
        {
            _tweetListener.Initialise(topic, e => _tweetProcessor.ProcessTweet(e));
        }

        // Start tweet listener
        // Get Events etc configured
        public void Start()
        {
            _tweetListener.Start();
        }
    }
}
