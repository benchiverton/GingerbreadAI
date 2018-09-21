namespace Emotion.Detector
{
    using Data;

    public static class EmotionTransformer
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
    }
}
