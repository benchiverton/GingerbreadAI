using System.Collections.Generic;
using System.Linq;

namespace Emotion.Detector.Lexicons.Extensions
{
    public static class StringExtensions
    {
        // this probably should be quite good
        // cba to remove diatrics
        public static List<string> GetWordsFromText(this string text)
        {
            var words = text.Split(' ').ToList();

            for (var i = 0; i < words.Count(); i++)
            {
                words[i] = string.Concat(words[i].Where(c => !char.IsPunctuation(c)));
            }

            words.ForEach(w => w = w.ToLower());

            return words;
        }
    }
}
