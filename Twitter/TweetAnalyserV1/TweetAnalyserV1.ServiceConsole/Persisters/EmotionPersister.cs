using System;
using System.Data;
using System.Data.SqlClient;
using Dapper;
using Emotion.Detector.Lexicons.Data;
using log4net;

namespace TweetAnalyserV1.ServiceConsole.Persisters
{
    public class EmotionPersister
    {
        private readonly ILog _log;
        private readonly string _connectionString;

        public EmotionPersister(ILog log)
        {
            _log = log;
            _connectionString = Environment.GetEnvironmentVariable("twitterRepositoryConnectionString");
        }

        public void PersistTweetEmotion(long tweetId, EmotionData tweetEmotion)
        {
            using (var dbConnection = new SqlConnection(_connectionString))
            {
                var spParameters = new DynamicParameters();
                spParameters.Add("@TweetId", tweetId);
                spParameters.Add("@Anger", tweetEmotion.Anger);
                spParameters.Add("@Anticipation", tweetEmotion.Anticipation);
                spParameters.Add("@Disgust", tweetEmotion.Disgust);
                spParameters.Add("@Fear", tweetEmotion.Fear);
                spParameters.Add("@Joy", tweetEmotion.Joy);
                spParameters.Add("@Negative", tweetEmotion.Negative);
                spParameters.Add("@Positive", tweetEmotion.Positive);
                spParameters.Add("@Sadness", tweetEmotion.Sadness);
                spParameters.Add("@Surprise", tweetEmotion.Surprise);
                spParameters.Add("@Trust", tweetEmotion.Trust);

                try
                {
                    dbConnection.Execute("[v1].[PersistTweetSentiment]", spParameters, commandType: CommandType.StoredProcedure);
                }
                catch (SqlException e) when (e.Number == 2627)
                {
                    _log.Warn($"Tweet with Id: {tweetId} has already been analysed & persisted.");
                }
            }
        }
    }
}
