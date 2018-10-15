using Emotion.Detector.Data;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace TwitterAnalyser.ServiceConsole.Persisters
{
    public class EmotionPersister
    {
        public Task PersistTweetEmotion(long tweetId, EmotionData tweetEmotion)
        {
            // persist the tweet to the db lol
        }
    }
}
