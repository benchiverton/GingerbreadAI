namespace Emotion.Detector.Extensions
{
    using System.Collections.Generic;
    using System.Linq;

    public static class StringExtensions
    {
        // this probaly should be quite good
        // cba to remove diatrics
        public static List<string> GetWordsFromText(this string text)
        {
            var words = text.Split(' ').ToList();

            for (var i = 0; i < words.Count(); i++)
            {
                words[i] = string.Concat(words[i].Where(c => !char.IsPunctuation(c)));
            }

            words.ForEach(w => w.ToLower());

            return words;
        }
    }
}
