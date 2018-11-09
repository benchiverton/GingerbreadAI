using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.Linq;
using Dapper;
using Dapper.FluentMap;
using Emotion.Detector.Lexicons.Data;
using Emotion.Detector.Lexicons.Mappers;
using Emotion.Detector.Lexicons.Repositories.Cache;

namespace Emotion.Detector.Lexicons.Repositories
{
    public class WordRepository
    {
        private readonly WordCache _cache;
        private readonly string _connectionString;

        public WordRepository(WordCache cache)
        {
            _cache = cache;
            _connectionString = Environment.GetEnvironmentVariable("wordRepositoryConnectionString");

            FluentMapper.Initialize(config =>
            {
                config.AddMap(new EmotionMap());
            });
        }

        public List<(string word, EmotionData emotion)> GetEmotions(List<string> words)
        {
            var emotions = new List<(string, EmotionData)>();
            foreach (var word in words)
            {
                if (!_cache.TryGetWordFromCache(word, out var emotion))
                {
                    if (TryGetEmotionFromDatabase(word, out emotion))
                    {
                        _cache.AddFoundWordToCache(word, emotion);
                    }
                    else
                    {
                        _cache.AddUnfoundWordToCache(word);
                    }
                }

                emotions.Add((word, emotion));
            }

            // emotionData is null if it is not found in cache or database.
            return emotions;
        }

        public bool TryGetEmotionFromDatabase(string word, out EmotionData emotionData)
        {
            using (var dbConnection = new SqlConnection(_connectionString))
            {
                var spParameters = new DynamicParameters();
                spParameters.Add("@Word", word);
                var resultList = dbConnection.Query<EmotionData>("[dbo].[GetSentimentFromWord]", spParameters, commandType: CommandType.StoredProcedure).ToList();

                if (resultList.Count != 0)
                {
                    emotionData = resultList.FirstOrDefault();
                    return true;
                }
                else
                {
                    emotionData = null;
                    return false;
                }
            }
        }
    }
}