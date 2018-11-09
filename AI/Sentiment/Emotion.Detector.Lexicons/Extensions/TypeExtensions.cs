using Newtonsoft.Json;

namespace Emotion.Detector.Lexicons.Extensions
{
    public static class TypeExtensions
    {
        public static T CloneJson<T>(this T source)
        {
            if (ReferenceEquals(source, null))
            {
                return default(T);
            }
            var deserializeSettings = new JsonSerializerSettings { ObjectCreationHandling = ObjectCreationHandling.Replace };
            return JsonConvert.DeserializeObject<T>(JsonConvert.SerializeObject(source), deserializeSettings);
        }
    }
}
