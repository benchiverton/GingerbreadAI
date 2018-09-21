namespace Emotion.Detector.Mapper
{
    using Dapper.FluentMap.Mapping;
    using Data;

    public class EmotionMap : EntityMap<Emotion>
    {
        public EmotionMap()
        {
            Map(i => i.Anger).ToColumn("Anger");
            Map(i => i.Disgust).ToColumn("Disgust");
            Map(i => i.Fear).ToColumn("Fear");
            Map(i => i.Happiness).ToColumn("Happiness");
            Map(i => i.Sadness).ToColumn("Sadness");
            Map(i => i.Surprise).ToColumn("Surprise");
        }
    }
}
