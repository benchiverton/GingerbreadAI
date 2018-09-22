namespace Emotion.Detector.Repository
{
    using System;
    using System.Collections.Generic;
    using System.Data;
    using System.Data.SqlClient;
    using System.Linq;
    using Cache;
    using Dapper;
    using Dapper.FluentMap;
    using Data;
    using Mapper;

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

        public List<(string word, Emotion emotion)> GetEmotions(List<string> words)
        {
            var emotions = new List<(string, Emotion)>();
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

            // emotion is null if it is not found in cache or database.
            return emotions;
        }

        public bool TryGetEmotionFromDatabase(string word, out Emotion emotion)
        {
            using (var dbConnection = new SqlConnection(_connectionString))
            {
                var spParameters = new DynamicParameters();
                spParameters.Add("@Word", word);
                var resultList = dbConnection.Query<Emotion>("[dbo].[GetEmotionsFromWord]", spParameters, commandType: CommandType.StoredProcedure).ToList();

                if (resultList.Count != 0)
                {
                    emotion = resultList.FirstOrDefault();
                    return true;
                }
                else
                {
                    emotion = null;
                    return false;
                }
            }
        }
    }
}