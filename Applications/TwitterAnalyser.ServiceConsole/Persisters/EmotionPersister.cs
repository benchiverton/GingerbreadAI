using Dapper;
using Emotion.Detector.Data;
using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.Threading.Tasks;

namespace TwitterAnalyser.ServiceConsole.Persisters
{
    public class EmotionPersister
    {
        private readonly string _connectionString;

        public EmotionPersister()
        {
            _connectionString = Environment.GetEnvironmentVariable("twitterRepositoryConnectionString", EnvironmentVariableTarget.User);
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

                dbConnection.Execute("[dbo].[PersistTweetSentiment]", spParameters, commandType: CommandType.StoredProcedure);
            }
        }
    }
}
