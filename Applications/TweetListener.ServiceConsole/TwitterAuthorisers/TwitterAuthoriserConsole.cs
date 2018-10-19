using System;

namespace TweetListener.ServiceConsole.TwitterAuthorisers
{
    public class TwitterAuthoriserConsole
    {
        public string GetPinCode(Uri authUrl)
        {
            System.Console.WriteLine($"Please visit the URL below, and enter the access code into the console.\r\n{authUrl.AbsoluteUri}");
            return System.Console.ReadLine();
        }
    }
}
