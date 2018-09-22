namespace Emotion.Detector.Mapper
{
    using Dapper.FluentMap.Mapping;
    using Data;

    public class EmotionMap : EntityMap<Emotion>
    {
        public EmotionMap()
        {
            Map(i => i.Anger).ToColumn("Anger");
            Map(i => i.Anticipation).ToColumn("Anticipation");
            Map(i => i.Disgust).ToColumn("Disgust");
            Map(i => i.Fear).ToColumn("Fear");
            Map(i => i.Joy).ToColumn("Joy");
            Map(i => i.Negative).ToColumn("Negative");
            Map(i => i.Positive).ToColumn("Positive");
            Map(i => i.Sadness).ToColumn("Sadness");
            Map(i => i.Surprise).ToColumn("Surprise");
            Map(i => i.Trust).ToColumn("Trust");
        }
    }
}
