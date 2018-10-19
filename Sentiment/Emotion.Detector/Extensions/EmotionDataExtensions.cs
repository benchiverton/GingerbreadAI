namespace Emotion.Detector.Extensions
{
    using Data;
    using System.Collections.Generic;
    using System.Text;

    public static class EmotionDataExtensions
    {
        public static void Invert(this EmotionData emotionData)
        {
            var properties = typeof(EmotionData).GetProperties();
            foreach (var property in properties)
            {
                var propertyValue = (float)property.GetValue(emotionData);
                if (propertyValue > 0.5)
                {
                    property.SetValue(emotionData, 1 - propertyValue);
                }
            }
        }

        public static string ToString(this EmotionData emotionData)
        {
            var str = new StringBuilder();
            var properties = typeof(EmotionData).GetProperties();
            foreach (var property in properties)
            {
                var propertyValue = (float)property.GetValue(emotionData);
                str.Append($"{property.Name}: {propertyValue}");
            }
            return str.ToString();
        }

        public static EmotionData GetOverallEmotion(this IEnumerable<EmotionData> emotions)
        {
            var overallEmotion = new EmotionData();
            foreach (var emotion in emotions)
            {
                var properties = typeof(EmotionData).GetProperties();
                foreach (var property in properties)
                {
                    property.SetValue(overallEmotion, (float)property.GetValue(overallEmotion) + (float)property.GetValue(emotion), null);
                }
            }
            return overallEmotion;
        }
    }
}
