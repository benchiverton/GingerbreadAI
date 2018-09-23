namespace Emotion.Detector.Extensions
{
    using Data;
    using System.Collections.Generic;
    using System.Text;

    public static class EmotionExtensions
    {
        public static void Invert(this Emotion emotion)
        {
            var properties = typeof(Emotion).GetProperties();
            foreach (var property in properties)
            {
                var propertyValue = (float)property.GetValue(emotion);
                if (propertyValue > 0.5)
                {
                    property.SetValue(emotion, 1 - propertyValue);
                }
            }
        }

        public static string ToString(this Emotion emotion)
        {
            var str = new StringBuilder();
            var properties = typeof(Emotion).GetProperties();
            foreach (var property in properties)
            {
                var propertyValue = (float)property.GetValue(emotion);
                str.Append($"{property.Name}: {propertyValue}");
            }
            return str.ToString();
        }

        public static Emotion GetOverallEmotion(this IEnumerable<Emotion> emotions)
        {
            var overallEmotion = new Emotion();
            foreach (var emotion in emotions)
            {
                var properties = typeof(Emotion).GetProperties();
                foreach (var property in properties)
                {
                    property.SetValue(overallEmotion, (float)property.GetValue(overallEmotion) + (float)property.GetValue(emotion), null);
                }
            }
            return overallEmotion;
        }
    }
}
