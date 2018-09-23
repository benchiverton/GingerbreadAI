using System;
using System.Collections.Generic;
using System.Text;

namespace TwitterProcessor.Console.TwitterAuthorisers
{
    public class TwitterAuthoriserConsole : ITwitterAuthoriser
    {
        public string GetPinCode(Uri authUrl)
        {
            System.Console.WriteLine($"Please visit the URL below, and enter the access code into the console.\r\n{authUrl.AbsoluteUri}");
            return System.Console.ReadLine();
        }
    }
}
